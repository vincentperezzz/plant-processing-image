"""Phone remote for the kiosk: live view, shutter, gallery, colour sliders.

Stdlib HTTP only — `http.server.ThreadingHTTPServer` with daemon threads. The
server never imports Tk and never touches a widget; everything it needs from
the kiosk arrives through the small `KioskBridge` interface below, which is
what makes it testable with a stub. `src.pi_sim.PiSim` implements that
interface with `remote_*` methods that marshal onto the Tk thread.

Security posture (chosen deliberately, see docs/install.md):
plain HTTP on the LAN plus a shared token. No TLS — a self-signed certificate
buys a browser warning and no real protection against anyone already on the
same network. Everything except the token-entry page requires the token, which
is accepted as a query parameter (an `<img src>` cannot send a header) or as a
header, and is always compared with `secrets.compare_digest`.

Image downloads are resolved through the scans database by integer id, and the
resulting path is verified to sit inside `SCANS_DIR` before a byte is served.
No client-supplied string is ever joined onto a filesystem path.
"""

from __future__ import annotations

import io
import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from src.frame_bus import ClientLimit, FrameBus
from src.paths import SCANS_DIR

BOUNDARY = b"plantframe"
MAX_BODY = 64 * 1024
DEFAULT_PORT = 8000
DEFAULT_MAX_CLIENTS = 3
SCAN_LIMIT = 200

# — kiosk palette, so the phone reads as the same product —
CREAM = "#f5ead8"
SURFACE = "#ebddc5"
TEXT = "#201e1d"
ACCENT = "#c67139"
ALERT = "#a52929"

# — self-hosted webfonts ————————————————————————————————————————————
# The phone may be on a LAN with no route to the internet, so the design
# system's Google Fonts @import cannot be used: it fails silently and the page
# quietly degrades to system faces. The same TTFs `pi_sim.py` loads through
# Pillow are served from here instead. The map is an allow-list of exact
# basenames — a client-supplied string is only ever compared against these
# keys, never joined onto a path.
FONTS_DIR = Path(__file__).resolve().parents[1] / "vendor" / "fonts"
FONT_FILES = (
    "Caprasimo-Regular.ttf",
    "Figtree-Regular.ttf",
    "Figtree-SemiBold.ttf",
    "Figtree-Bold.ttf",
)
FONT_CACHE = "public, max-age=31536000, immutable"

OPEN_NOTE = (
    "Open mode: no token is in force. Anyone on this network who knows the "
    "address can watch and control this kiosk. Traffic is not encrypted."
)
TOKEN_NOTE = (
    "Local network only, and not encrypted. Anyone with the token and the "
    "address can watch and control this kiosk."
)


class KioskBridge(Protocol):
    """The whole of what the server is allowed to know about the kiosk.

    Every method is called from an HTTP worker thread. Implementations are
    responsible for marshalling onto their own UI thread.
    """

    def remote_shutter(self) -> dict:
        """Try a capture. -> {"ok": bool, "reason": str}. Honest about busy."""

    def remote_color(self) -> dict:
        """-> {"profile": {...}, "ranges": {...}, "active": str, "native": bool,
        "sliders": [{"name","label"}, ...]}"""

    def remote_set_slider(self, name: str, value: float) -> dict:
        """Move one slider (clamped to its range) and return `remote_color()`."""

    def remote_profile(self, action: str, name: str | None = None) -> dict:
        """action in {"activate", "save", "reset"}. Returns `remote_color()`."""

    def remote_hud(self) -> dict:
        """Live HUD text: crop, health (key + label), notes, tone, confidence."""

    def remote_gallery_changed(self) -> None:
        """Told after a remote delete so the kiosk list can catch up."""


# — helpers ————————————————————————————————————————————————————————


def lan_ip() -> str:
    """Best guess at the address a phone should type. No traffic is sent.

    A UDP socket has no handshake, so `connect()` only asks the routing table
    which local address would be used. Falls back to loopback with no network.
    """
    for probe in (("8.8.8.8", 53), ("192.0.2.1", 9)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(probe)
            addr = sock.getsockname()[0]
            if addr and not addr.startswith("0."):
                return str(addr)
        except OSError:
            continue
        finally:
            sock.close()
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _scan_row(scan_id: int) -> dict | None:
    from src.records import list_scans

    for row in list_scans(limit=SCAN_LIMIT):
        if int(row.get("id") or 0) == scan_id:
            return row
    return None


def safe_scan_path(image_path) -> Path | None:
    """Resolve a stored image path, or None if it escapes SCANS_DIR.

    Both sides are resolved before the comparison so a symlink or a `..` that
    survived into the database cannot walk out of the scans directory.
    """
    if not image_path:
        return None
    try:
        path = Path(str(image_path)).resolve()
        root = Path(SCANS_DIR).resolve()
    except OSError:
        return None
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


class _Ctx:
    """Everything a handler needs, hung off the server instance."""

    def __init__(self, bridge, bus, token, max_clients, stream_width, open_mode=False):
        self.bridge = bridge
        self.bus = bus
        self.token = token
        self.max_clients = max_clients
        self.stream_width = stream_width
        self.open_mode = bool(open_mode)
        self.stopping = False


# — request handler ————————————————————————————————————————————————


class _Handler(BaseHTTPRequestHandler):
    server_version = "PlantHealthKiosk/1.0"
    protocol_version = "HTTP/1.1"

    # The kiosk console belongs to the operator; one line per request is noise.
    def log_message(self, fmt, *args) -> None:  # noqa: A003
        pass

    # — plumbing —

    @property
    def ctx(self) -> _Ctx:
        return self.server.ctx

    def _split(self) -> tuple[str, dict]:
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    def _authed(self, query: dict) -> bool:
        if self.ctx.open_mode:
            return True
        want = self.ctx.token
        given = ""
        values = query.get("token") or []
        if values:
            given = str(values[0])
        if not given:
            given = self.headers.get("X-Auth-Token", "") or ""
        if not given:
            auth = self.headers.get("Authorization", "") or ""
            if auth.lower().startswith("bearer "):
                given = auth[7:].strip()
        if not given:
            return False
        return secrets.compare_digest(str(given), str(want))

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str,
        extra: dict | None = None,
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _deny(self) -> None:
        self._json({"ok": False, "error": "bad or missing token"}, 401)

    def _body(self) -> bytes | None:
        """Bounded read. None means the caller already got an error response."""
        if (self.headers.get("Transfer-Encoding", "") or "").lower() == "chunked":
            self._json({"ok": False, "error": "chunked bodies not accepted"}, 411)
            return None
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json({"ok": False, "error": "bad Content-Length"}, 400)
            return None
        if length < 0 or length > MAX_BODY:
            self._json({"ok": False, "error": "body too large"}, 413)
            return None
        return self.rfile.read(length) if length else b""

    @staticmethod
    def _int_arg(query: dict, name: str) -> int | None:
        values = query.get(name) or []
        if not values:
            return None
        try:
            return int(str(values[0]))
        except (TypeError, ValueError):
            return None

    # — verbs —

    def do_GET(self) -> None:  # noqa: N802
        path, query = self._split()
        if path == "/":
            self._page(query)
            return
        if path.startswith("/fonts/"):
            # Fonts are not secret and a stylesheet's url() carries no token,
            # so this route sits in front of the auth gate.
            self._font(path[len("/fonts/") :])
            return
        if not self._authed(query):
            self._deny()
            return
        if path == "/stream.mjpg":
            self._stream()
        elif path == "/api/scans":
            self._scans()
        elif path == "/api/color":
            self._color_get()
        elif path == "/api/status":
            self._status()
        elif path in ("/photo", "/thumb"):
            self._image(query, thumb=(path == "/thumb"))
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path, query = self._split()
        if not self._authed(query):
            self._deny()
            return
        body = self._body()
        if body is None:
            return
        if path == "/shutter":
            self._shutter()
        elif path in ("/api/retry", "/api/rescan"):
            self._retry_post()
        elif path == "/api/tap":
            self._tap_post(body)
        elif path == "/api/color":
            self._color_post(body)
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_DELETE(self) -> None:  # noqa: N802
        path, query = self._split()
        if not self._authed(query):
            self._deny()
            return
        if self._body() is None:
            return
        if path == "/api/scan":
            self._delete(query)
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    # — routes —

    def _page(self, query: dict) -> None:
        if self._authed(query):
            note = OPEN_NOTE if self.ctx.open_mode else TOKEN_NOTE
            html = PAGE_HTML.replace("__TOKEN__", "" if self.ctx.open_mode else self.ctx.token)
            html = html.replace("__ACCESS_NOTE__", note)
        else:
            html = LOGIN_HTML
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _font(self, name: str) -> None:
        """Serve one vendored TTF, chosen from a fixed allow-list.

        `name` is only ever compared against the literals in `FONT_FILES`; the
        path is then built from the matching literal, so no client-supplied
        string reaches the filesystem.
        """
        chosen = None
        for allowed in FONT_FILES:
            if name == allowed:
                chosen = allowed
                break
        if chosen is None:
            self._json({"ok": False, "error": "not found"}, 404)
            return
        try:
            data = (FONTS_DIR / chosen).read_bytes()
        except OSError:
            self._json({"ok": False, "error": "font unavailable"}, 404)
            return
        self._send(200, data, "font/ttf", cache=FONT_CACHE)

    def _color_get(self) -> None:
        try:
            state = self.ctx.bridge.remote_color() or {}
        except Exception as exc:
            self._json({"ok": False, "error": f"kiosk did not answer: {exc}"}, 503)
            return
        self._json({"ok": True, **state})

    def _status(self) -> None:
        bus = self.ctx.bus
        payload = {
            "ok": True,
            "seq": bus.seq if bus else 0,
            "clients": bus.client_count if bus else 0,
            "frame_age": round(bus.age(), 2) if bus else 1e9,
            "max_clients": self.ctx.max_clients,
        }
        # The HUD is a bonus on top of the liveness fields: if the kiosk cannot
        # answer, the page must still learn that the frame has gone stale, so a
        # missing HUD is an omission and never an error.
        try:
            hud = self.ctx.bridge.remote_hud() or {}
        except Exception:
            hud = {}
        payload["hud"] = {
            "crop": str(hud.get("crop") or ""),
            "health": str(hud.get("health") or ""),
            "health_label": str(hud.get("health_label") or ""),
            "notes": str(hud.get("notes") or ""),
            "notes_extra": str(hud.get("notes_extra") or ""),
            "tone": str(hud.get("tone") or "muted"),
            "confidence": hud.get("confidence"),
        }
        self._json(payload)

    def _shutter(self) -> None:
        try:
            result = self.ctx.bridge.remote_shutter() or {}
        except Exception as exc:
            self._json({"ok": False, "reason": f"kiosk did not answer: {exc}"}, 503)
            return
        self._json({"ok": bool(result.get("ok")), "reason": result.get("reason", "")})

    def _scans(self) -> None:
        from src.records import list_scans

        out = []
        for row in list_scans(limit=SCAN_LIMIT):
            if safe_scan_path(row.get("image_path")) is None:
                continue
            out.append(
                {
                    "id": int(row.get("id") or 0),
                    "created_at": row.get("created_at") or "",
                    "crop": row.get("crop") or "",
                    "health": row.get("health") or "",
                    "named_plant": row.get("named_plant") or "",
                    "tip": row.get("tip") or "",
                    "confidence": row.get("confidence"),
                }
            )
        self._json({"ok": True, "scans": out})

    def _image(self, query: dict, *, thumb: bool) -> None:
        scan_id = self._int_arg(query, "id")
        if scan_id is None:
            self._json({"ok": False, "error": "id required"}, 400)
            return
        row = _scan_row(scan_id)
        if row is None:
            self._json({"ok": False, "error": "no such scan"}, 404)
            return
        path = safe_scan_path(row.get("image_path"))
        if path is None:
            # Either the file is gone or it sits outside SCANS_DIR. Both are a
            # flat refusal; we never serve a path we did not vouch for.
            self._json({"ok": False, "error": "image unavailable"}, 404)
            return
        if thumb:
            data = _thumb_bytes(path)
            if data is None:
                self._json({"ok": False, "error": "image unavailable"}, 404)
                return
            self._send(200, data, "image/jpeg")
            return
        try:
            data = path.read_bytes()
        except OSError:
            self._json({"ok": False, "error": "image unavailable"}, 404)
            return
        extra = {}
        if (query.get("download") or []) and str((query.get("download") or [""])[0]) not in ("", "0"):
            extra["Content-Disposition"] = f'attachment; filename="scan-{scan_id}{path.suffix}"'
        ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        self._send(200, data, ctype, extra)

    def _delete(self, query: dict) -> None:
        from src.records import delete_scan

        scan_id = self._int_arg(query, "id")
        confirm = str((query.get("confirm") or [""])[0]).lower()
        if scan_id is None:
            self._json({"ok": False, "error": "id required"}, 400)
            return
        if confirm not in ("1", "yes", "true"):
            self._json({"ok": False, "error": "confirm=yes required"}, 400)
            return
        row = _scan_row(scan_id)
        if row is None:
            self._json({"ok": False, "error": "no such scan"}, 404)
            return
        path = safe_scan_path(row.get("image_path"))
        delete_scan(scan_id, str(path) if path is not None else "")
        try:
            self.ctx.bridge.remote_gallery_changed()
        except Exception:
            pass
        self._json({"ok": True, "deleted": scan_id})

    def _color_post(self, body: bytes) -> None:
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            self._json({"ok": False, "error": "bad json"}, 400)
            return
        if not isinstance(data, dict):
            self._json({"ok": False, "error": "bad json"}, 400)
            return
        action = str(data.get("action") or "set")
        bridge = self.ctx.bridge
        try:
            if action == "set":
                name = str(data.get("name") or "")
                value = float(data.get("value"))
                state = bridge.remote_set_slider(name, value)
            elif action in ("activate", "save"):
                name = str(data.get("name") or "")
                state = bridge.remote_profile(action, name)
            elif action == "reset":
                state = bridge.remote_profile("reset", None)
            else:
                self._json({"ok": False, "error": "unknown action"}, 400)
                return
        except (KeyError, ValueError, TypeError) as exc:
            self._json({"ok": False, "error": f"bad request: {exc}"}, 400)
            return
        except Exception as exc:
            self._json({"ok": False, "error": f"kiosk did not answer: {exc}"}, 503)
            return
        self._json({"ok": True, **(state or {})})

    def _retry_post(self) -> None:
        try:
            res = self.ctx.bridge.remote_retry()
            self._json(res or {"ok": True})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 503)

    def _tap_post(self, body: bytes) -> None:
        try:
            data = json.loads(body.decode("utf-8") or "{}")
            x = float(data.get("x", 0.5))
            y = float(data.get("y", 0.5))
            res = self.ctx.bridge.remote_tap(x, y)
            self._json(res or {"ok": True})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)

    def _stream(self) -> None:
        bus = self.ctx.bus
        if bus is None:
            self._json({"ok": False, "error": "no stream"}, 503)
            return
        try:
            cm = bus.client(self.ctx.max_clients)
            cm.__enter__()
        except ClientLimit:
            self._json(
                {"ok": False, "error": f"too many viewers (max {self.ctx.max_clients})"}, 503
            )
            return
        try:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode("ascii"),
            )
            self.end_headers()
            self.close_connection = True
            last = 0
            idle = 0
            while not self.ctx.stopping:
                got = bus.wait(last, timeout=1.0)
                if got is None:
                    idle += 1
                    if idle > 60:  # a minute with no frames: let the client retry
                        break
                    continue
                idle = 0
                last, jpeg = got
                self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError, ValueError):
            pass
        finally:
            cm.__exit__(None, None, None)


def _thumb_bytes(path: Path, size: int = 320) -> bytes | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=78)
            return buf.getvalue()
    except Exception:
        return None


# — server ——————————————————————————————————————————————————————————


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class KioskServer:
    """Owns the socket and the serving thread. Nothing is bound until start()."""

    def __init__(
        self,
        bridge: KioskBridge,
        *,
        bus: FrameBus,
        port: int = DEFAULT_PORT,
        token: str | None = None,
        host: str = "0.0.0.0",
        max_clients: int = DEFAULT_MAX_CLIENTS,
        stream_width: int = 1024,
        open_mode: bool = False,
    ) -> None:
        self.bridge = bridge
        self.bus = bus
        self.host = host
        self.port = int(port)
        self.token = str(token) if token else secrets.token_urlsafe(8)
        self.max_clients = int(max_clients)
        self.stream_width = int(stream_width)
        self.open_mode = bool(open_mode)
        self._httpd: _HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def url(self, host: str | None = None) -> str:
        if self.open_mode:
            return f"http://{host or lan_ip()}:{self.port}/"
        return f"http://{host or lan_ip()}:{self.port}/?token={self.token}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        httpd = _HTTPServer((self.host, self.port), _Handler)
        httpd.ctx = _Ctx(
            self.bridge,
            self.bus,
            self.token,
            self.max_clients,
            self.stream_width,
            open_mode=self.open_mode,
        )
        self._httpd = httpd
        # Port 0 means "pick one"; report what was actually bound.
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Call from the Tk thread, never from a serving thread (deadlock)."""
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        if httpd is None:
            return
        try:
            httpd.ctx.stopping = True
        except AttributeError:
            pass
        try:
            httpd.shutdown()
        except Exception:
            pass
        if thread is not None:
            # Bounded: a wedged client must not be able to hang kiosk exit.
            thread.join(timeout=2.0)
        try:
            httpd.server_close()
        except Exception:
            pass


# — the page ————————————————————————————————————————————————————————
#
# One self-contained document: every byte of CSS and JS is inline, and the only
# sub-resources the page asks for are this server's own font, stream, thumbnail
# and API routes. Nothing is fetched from the internet.
#
# The look is the project's design system (docs/UI-modernization/reference/
# design-tokens.css) used directly — real tokens, Caprasimo/Figtree, the card,
# pill, LIVE-badge and health-tag idioms — but the *geometry* is re-laid-out for
# a ~390px portrait phone rather than the kiosk's 1024x600 landscape panel:
# one column, full-bleed cards, and the shutter parked in a fixed bottom dock
# where a thumb rests.

FONT_CSS = """
 @font-face{font-family:"Caprasimo";font-style:normal;font-weight:400;font-display:swap;
   src:url("/fonts/Caprasimo-Regular.ttf") format("truetype")}
 @font-face{font-family:"Figtree";font-style:normal;font-weight:400;font-display:swap;
   src:url("/fonts/Figtree-Regular.ttf") format("truetype")}
 @font-face{font-family:"Figtree";font-style:normal;font-weight:600;font-display:swap;
   src:url("/fonts/Figtree-SemiBold.ttf") format("truetype")}
 @font-face{font-family:"Figtree";font-style:normal;font-weight:700;font-display:swap;
   src:url("/fonts/Figtree-Bold.ttf") format("truetype")}
"""

TOKENS_CSS = """
 :root{
   color-scheme:light;
   --color-bg:#f5ead8; --color-surface:#ebddc5; --color-text:#201e1d;
   --color-accent:#c67139; --color-accent-2:#7a8a5e; --color-alert:#a52929;
   --color-divider:rgba(32,30,29,.16);
   --color-neutral-100:#f9f4ed; --color-neutral-300:#dcd3c4; --color-neutral-400:#c0b6a5;
   --color-neutral-800:#474238; --color-neutral-900:#2e2b25;
   --color-accent-100:#fff2eb; --color-accent-600:#b2622d; --color-accent-700:#8c491a;
   --color-accent-800:#643312;
   --color-accent-2-100:#f0fae1; --color-accent-2-800:#3d472b;
   --font-heading:"Caprasimo",system-ui,sans-serif;
   --font-body:"Figtree",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
   --space-1:4.4px; --space-2:8.8px; --space-3:13.2px; --space-4:17.6px;
   --space-6:26.4px; --space-8:35.2px;
   --radius-sm:8px; --radius-md:16px; --radius-lg:28px;
   --shadow-sm:0 1px 2px rgba(46,43,37,.14);
   --shadow-md:0 3px 10px rgba(46,43,37,.16);
   --shadow-lg:0 12px 32px rgba(46,43,37,.22);
   /* health grades, straight from the handoff table */
   --grade-healthy-bg:#cdf0cd; --grade-healthy-fg:#004906;
   --grade-mild-bg:#fae2b0;    --grade-mild-fg:#6e3800;
   --grade-critical-bg:#ffe4e1;--grade-critical-fg:#83000d;
   --grade-dead-bg:#dcd3c4;    --grade-dead-fg:#2e2b25;
 }
"""

LOGIN_HTML = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Plant Health</title>
<style>"""
    + FONT_CSS
    + TOKENS_CSS
    + """
 *,*::before,*::after{box-sizing:border-box}
 body{margin:0;background:var(--color-bg);color:var(--color-text);
      font:400 15px/1.55 var(--font-body);
      display:flex;min-height:100vh;align-items:center;justify-content:center;padding:var(--space-6)}
 form{background:var(--color-surface);border-radius:calc(var(--radius-lg) * 1.15);
      padding:var(--space-6);width:min(420px,100%);box-shadow:var(--shadow-md)}
 h1{font:400 25px/1.12 var(--font-heading);letter-spacing:-.015em;margin:0 0 var(--space-1)}
 p{margin:0 0 var(--space-4);font-size:13px;
   color:color-mix(in srgb,var(--color-text) 60%,transparent)}
 input{width:100%;min-height:52px;font:inherit;font-size:17px;padding:0 16px;border-radius:999px;
       border:1px solid var(--color-divider);background:var(--color-bg);color:var(--color-text);
       caret-color:var(--color-accent)}
 input:focus-visible{outline:2px solid var(--color-accent);outline-offset:2px}
 button{margin-top:var(--space-3);width:100%;min-height:52px;font:400 17px/1.2 var(--font-heading);
        border:0;border-radius:999px;background:var(--color-accent);color:var(--color-bg)}
 button:active{background:var(--color-accent-700)}
</style></head><body>
<form onsubmit="location='/?token='+encodeURIComponent(this.t.value.trim());return false">
 <h1>Plant Health remote</h1>
 <p>Enter the access token shown on the kiosk Settings page.</p>
 <input name="t" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="token">
 <button type="submit">Connect</button>
</form></body></html>
"""
)

PAGE_HTML = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Plant Health remote</title>
<style>"""
    + FONT_CSS
    + TOKENS_CSS
    + """
 *,*::before,*::after{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 /* The dock's gradient fades to the page colour, so anything showing through
    under a short page has to be the page colour too. */
 html{background:var(--color-bg)}
 body{margin:0;background:var(--color-bg);color:var(--color-text);
      min-height:100vh;
      font:400 15px/1.55 var(--font-body);
      padding:0 var(--space-3) calc(112px + env(safe-area-inset-bottom));
      max-width:640px;margin:0 auto;transition:max-width .2s ease,padding .2s ease}
 body.cam{padding-bottom:calc(198px + env(safe-area-inset-bottom))}
 [hidden]{display:none !important}
 h1,h2,h3{font-family:var(--font-heading);font-weight:400;line-height:1.12;
          letter-spacing:-.015em;margin:0}
 .muted{color:color-mix(in srgb,var(--color-text) 55%,transparent)}
 :focus-visible{outline:2px solid var(--color-accent);outline-offset:2px}

 /* — top bar — */
 .topbar{display:flex;align-items:center;gap:var(--space-2);
         padding:var(--space-3) var(--space-1) var(--space-2)}
 .topbar svg{display:block;flex:none}
 .brand{font-family:var(--font-heading);font-weight:400;font-size:20px;margin-right:auto}
 .conn{font-size:11px;letter-spacing:.02em;padding:4px 11px;border-radius:999px;
       background:var(--color-accent-2-100);color:var(--color-accent-2-800);white-space:nowrap;
       transition:background .2s ease,color .2s ease}
 .conn.bad{background:#ffe4e1;color:#83000d}
 .conn.idle{background:var(--color-neutral-100);color:var(--color-neutral-800)}

 /* — viewfinder — */
 .stage{position:relative;background:var(--color-neutral-900);
        border-radius:var(--radius-lg);overflow:hidden;
        aspect-ratio:16/9;width:100%;
        box-shadow:var(--shadow-md);border:1px solid var(--color-divider);
        display:flex;align-items:center;justify-content:center;transition:box-shadow .2s ease}
 .stage img{width:100%;height:100%;object-fit:cover;display:block}
 .live{position:absolute;left:14px;top:14px;display:flex;align-items:center;gap:6px;
       background:var(--color-alert);color:#fff;font-size:12px;font-weight:700;
       padding:6px 12px;border-radius:999px;letter-spacing:.04em;box-shadow:var(--shadow-sm);
       backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:2}
 .live i{width:7px;height:7px;border-radius:50%;background:#fff;display:block;
         animation:pulse 1.6s ease-in-out infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
 .veil{position:absolute;inset:0;display:none;flex-direction:column;gap:6px;
       align-items:center;justify-content:center;text-align:center;padding:var(--space-4);
       background:rgba(46,43,37,.82);color:var(--color-neutral-100);backdrop-filter:blur(6px);
       -webkit-backdrop-filter:blur(6px);z-index:3}
 .veil b{font-family:var(--font-heading);font-weight:400;font-size:19px}
 .veil span{font-size:12px;color:var(--color-neutral-400)}
 body.stale .veil{display:flex}
 body.stale .live{background:var(--color-neutral-800)}
 body.stale .live i{animation:none;opacity:.5}

 /* — status cards — */
 .hud{display:grid;grid-template-columns:1fr 1fr;gap:var(--space-2);
      margin-top:var(--space-3)}
 .hcard{background:var(--color-surface);border-radius:var(--radius-lg);
        padding:var(--space-3) var(--space-4);box-shadow:var(--shadow-sm);
        border:1px solid var(--color-divider);min-height:88px;
        transition:background .2s ease,color .2s ease,box-shadow .2s ease,border-color .2s ease}
 .hcard.wide{grid-column:1 / -1;min-height:0}
 .hcard .k{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
           opacity:.62;font-weight:600}
 .hcard .v{font-family:var(--font-heading);font-weight:400;font-size:20px;
           line-height:1.2;margin-top:3px;word-break:break-word}
 .hcard.wide .v{font-family:var(--font-body);font-size:14.5px;line-height:1.45}
 .hcard.wide .v em{font-style:normal;opacity:.55}
 .hcard.warn .v{color:var(--color-alert);font-weight:600}
 .hcard.healthy{background:var(--grade-healthy-bg);color:var(--grade-healthy-fg);border-color:color-mix(in srgb,var(--grade-healthy-fg) 25%,transparent)}
 .hcard.mild{background:var(--grade-mild-bg);color:var(--grade-mild-fg);border-color:color-mix(in srgb,var(--grade-mild-fg) 25%,transparent)}
 .hcard.critical{background:var(--grade-critical-bg);color:var(--grade-critical-fg);border-color:color-mix(in srgb,var(--grade-critical-fg) 25%,transparent)}
 .hcard.dead{background:var(--grade-dead-bg);color:var(--grade-dead-fg);border-color:color-mix(in srgb,var(--grade-dead-fg) 25%,transparent)}
 .conf{display:inline-flex;align-items:center;margin-top:7px;padding:2px 10px;
       border:2px solid currentColor;border-radius:999px;
       font:700 10.5px/1.4 var(--font-body);letter-spacing:.04em}

 /* — sections — */
 .sect{display:flex;align-items:baseline;gap:var(--space-2);
       margin:0 var(--space-1) var(--space-2)}
 .sect h2{font-size:22px}
 .sect .hint{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
             margin-left:auto;color:color-mix(in srgb,var(--color-text) 50%,transparent)}
 .card{background:var(--color-surface);border-radius:calc(var(--radius-lg) * 1.15);
       padding:var(--space-4);box-shadow:var(--shadow-sm);border:1px solid var(--color-divider)}

 /* — sliders — */
 .row{display:grid;grid-template-columns:1fr auto;align-items:center;
      gap:0 var(--space-2);padding:var(--space-1) 0}
 .row label{font-size:13px;font-weight:600}
 .row output{font-size:13px;font-variant-numeric:tabular-nums;
             padding:2px 9px;border-radius:999px;background:transparent;
             color:color-mix(in srgb,var(--color-text) 60%,transparent);
             transition:background .18s ease,color .18s ease}
 .row output.pending{background:var(--color-accent-100);color:var(--color-accent-800);
                     font-weight:700}
 .row output.saved{background:var(--color-accent-2-100);color:var(--color-accent-2-800);
                   font-weight:700}
 .row input[type=range]{grid-column:1 / -1;width:100%;height:44px;margin:0;
                        accent-color:var(--color-accent);background:transparent;
                        -webkit-appearance:none;appearance:none;touch-action:pan-y}
 .row input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:999px;
   background:color-mix(in srgb,var(--color-text) 14%,transparent)}
 .row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
   width:26px;height:26px;margin-top:-10px;border-radius:50%;
   background:var(--color-accent);border:3px solid var(--color-surface);
   box-shadow:var(--shadow-sm);transition:transform .12s ease}
 .row input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15)}
 .row input[type=range]::-moz-range-track{height:6px;border-radius:999px;
   background:color-mix(in srgb,var(--color-text) 14%,transparent)}
 .row input[type=range]::-moz-range-thumb{width:22px;height:22px;border-radius:50%;
   background:var(--color-accent);border:3px solid var(--color-surface)}

 /* — pills — */
 .pills{display:flex;flex-wrap:wrap;gap:var(--space-2);margin-top:var(--space-3)}
 .pill{flex:1 1 30%;min-height:46px;padding:var(--space-2) var(--space-3);
       border-radius:999px;border:1px solid var(--color-divider);background:transparent;
       color:var(--color-text);font:400 15px/1.2 var(--font-heading);cursor:pointer;
       display:inline-flex;align-items:center;justify-content:center;
       transition:background .15s ease,border-color .15s ease,color .15s ease,transform .12s ease}
 .pill:active{background:rgba(32,30,29,.10);transform:scale(.98)}
 .pill.on{background:var(--color-accent);border-color:var(--color-accent);
          color:var(--color-bg)}
 .pill.sub{font-size:13px;min-height:44px;flex:1 1 45%}

 /* — gallery — */
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
       gap:var(--space-3)}
 .tile{background:var(--color-surface);border-radius:var(--radius-lg);overflow:hidden;
       box-shadow:var(--shadow-sm);border:1px solid var(--color-divider);display:flex;flex-direction:column;
       padding:0 0 var(--space-3);border:1px solid var(--color-divider);text-align:left;color:inherit;
       font:inherit;cursor:pointer;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease}
 .tile:hover{transform:translateY(-3px);box-shadow:var(--shadow-md);border-color:var(--color-accent)}
 .tile:active{background:var(--color-neutral-300);transform:translateY(0)}
 .tile img{width:100%;aspect-ratio:1;object-fit:cover;display:block;
           background:var(--color-neutral-900)}
 .tile .meta{padding:var(--space-2) var(--space-3) 0}
 .tile .name{font-family:var(--font-heading);font-weight:400;font-size:17px;line-height:1.2}
 .tile .when{font-size:11px;margin-top:2px;
             color:color-mix(in srgb,var(--color-text) 50%,transparent)}
 .tag{display:inline-flex;align-items:center;font-size:11px;letter-spacing:.02em;
      padding:3px 10px;border-radius:999px;margin-top:6px;
      background:var(--color-neutral-100);color:var(--color-neutral-800)}
 .tag.healthy{background:var(--grade-healthy-bg);color:var(--grade-healthy-fg)}
 .tag.mild{background:var(--grade-mild-bg);color:var(--grade-mild-fg)}
 .tag.critical{background:var(--grade-critical-bg);color:var(--grade-critical-fg)}
 .tag.dead{background:var(--grade-dead-bg);color:var(--grade-dead-fg)}
 .empty{grid-column:1 / -1;padding:var(--space-6);text-align:center;font-size:14px;
        border:1px dashed var(--color-divider);border-radius:var(--radius-lg);
        color:color-mix(in srgb,var(--color-text) 55%,transparent)}

 /* — one scan, full size — */
 .back{display:inline-flex;align-items:center;gap:8px;min-height:44px;
       padding:0 18px 0 14px;border-radius:999px;border:1px solid var(--color-divider);
       background:transparent;color:var(--color-text);
       font:400 15px/1.2 var(--font-heading);cursor:pointer;margin-bottom:var(--space-3);
       transition:background .15s ease,border-color .15s ease}
 .back:hover{background:rgba(32,30,29,.06);border-color:var(--color-accent)}
 .back:active{background:rgba(32,30,29,.10)}
 .shot{width:100%;height:auto;display:block;border-radius:var(--radius-lg);
       background:var(--color-neutral-900);box-shadow:var(--shadow-md);border:1px solid var(--color-divider)}
 .dmeta{margin-top:var(--space-3)}
 .dmeta .name{font-family:var(--font-heading);font-size:24px;line-height:1.15}
 .dmeta .when{font-size:12px;margin-top:4px;
              color:color-mix(in srgb,var(--color-text) 55%,transparent)}
 .dnotes{margin-top:var(--space-3);background:var(--color-surface);
         border-radius:var(--radius-lg);padding:var(--space-3) var(--space-4);
         font-size:14px;box-shadow:var(--shadow-sm);border:1px solid var(--color-divider)}
 .acts{display:flex;gap:var(--space-2);margin-top:var(--space-3)}
 .acts a,.acts button{flex:1;min-height:48px;display:inline-flex;align-items:center;
   justify-content:center;border-radius:999px;border:1px solid var(--color-divider);
   background:transparent;color:var(--color-text);font:400 15px/1.2 var(--font-heading);
   text-decoration:none;cursor:pointer;transition:background .15s ease,border-color .15s ease}
 .acts a:hover{background:rgba(32,30,29,.06);border-color:var(--color-accent)}
 .acts button{color:var(--color-alert);
              border-color:color-mix(in srgb,var(--color-alert) 40%,transparent)}
 .acts button:hover{background:rgba(165,41,41,.08)}
 .acts button.armed{background:var(--color-alert);border-color:var(--color-alert);color:#fff}

 .note{font-size:11px;line-height:1.5;margin:var(--space-6) var(--space-1) 0;
       color:color-mix(in srgb,var(--color-text) 50%,transparent)}

 /* — bottom dock — */
 .dock{position:fixed;left:0;right:0;bottom:0;z-index:5;
       display:flex;flex-direction:column;align-items:center;gap:var(--space-2);
       padding:var(--space-2) var(--space-3) calc(var(--space-2) + env(safe-area-inset-bottom));
       background:linear-gradient(to bottom,rgba(245,234,216,0) 0%,var(--color-bg) 30%)}
 .dock > *{width:100%;max-width:614px}
 .toast{min-height:17px;font-size:12px;text-align:center;
        color:color-mix(in srgb,var(--color-text) 62%,transparent);
        transition:color .18s ease}
 .toast.bad{color:var(--color-alert);font-weight:600}
 .shutter{width:76px;height:76px;border-radius:50%;padding:0;cursor:pointer;
          background:var(--color-bg);border:5px solid var(--color-accent);
          display:grid;place-items:center;box-shadow:var(--shadow-md);
          flex:none;margin:0 auto;transition:transform .15s ease,box-shadow .15s ease}
 .shutter::after{content:"";width:52px;height:52px;border-radius:50%;
                 background:var(--color-accent);transition:transform .12s ease}
 .shutter:hover:not([disabled]){transform:scale(1.05);box-shadow:var(--shadow-lg)}
 .shutter:active::after{transform:scale(.86)}
 .shutter[disabled]{opacity:.5;cursor:not-allowed}
 .tabs{display:flex;gap:6px;padding:5px;border-radius:999px;
       background:var(--color-surface);box-shadow:var(--shadow-sm);border:1px solid var(--color-divider)}
 .tabs .pill{flex:1 1 0;min-height:46px;font-size:15px;border:0;background:transparent}
 .tabs .pill[aria-selected="true"]{background:var(--color-accent);color:var(--color-bg);
                                   box-shadow:var(--shadow-sm)}
 .tabs .pill:hover:not([aria-selected="true"]){background:rgba(32,30,29,.06)}

 /* — Settings Mobile Base — */
 #view-settings{display:flex;flex-direction:column;gap:var(--space-3)}
 #view-settings .stage{aspect-ratio:16/9;width:100%;border-radius:var(--radius-lg);overflow:hidden;
                       box-shadow:var(--shadow-md);border:1px solid var(--color-divider);position:relative;background:var(--color-neutral-900)}
 #view-settings .stage img{width:100%;height:100%;object-fit:cover;display:block}
 .settings-header{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:var(--space-1)}
 .settings-header h2{font-size:20px}
 .settings-header .hint{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 50%,transparent)}
 .settings-side{display:flex;flex-direction:column;gap:var(--space-2);background:var(--color-bg);
                padding:var(--space-2) var(--space-3);border-radius:var(--radius-md);border:1px solid var(--color-divider);margin-top:var(--space-2)}

 /* — 16:9 PC widescreen / Desktop responsive layout — */
 @media (min-width: 860px){
   body{max-width:1440px;width:94vw;min-height:100vh;margin:0 auto;
        padding:var(--space-2) var(--space-5) 120px;box-sizing:border-box}
   body.cam{padding-bottom:120px}
   .topbar{padding:12px 0;border-bottom:1px solid var(--color-divider);
           margin-bottom:0;display:flex;align-items:center;justify-content:space-between;width:100%}
   .brand{font-size:26px}
   .conn{font-size:12px;padding:6px 16px}
   main{width:100%}

   /* Camera & Settings Tabs: 100% IDENTICAL vertical height, top distance (168px), and 16:9 geometry */
   #view-camera, #view-settings{
     display:grid;grid-template-columns:minmax(0,1.65fr) minmax(360px,1fr);
     gap:32px;align-items:stretch;width:100%;
     margin-top:clamp(24px, calc((100vh - (min(1440px, 94vw) * 0.62 * 9 / 16) - 170px) / 2), 168px);
     margin-bottom:auto}
   #view-camera .stage, #view-settings .stage{
     aspect-ratio:16/9;width:100%;height:100%;min-height:0;
     border-radius:24px;box-shadow:var(--shadow-md);
     border:1px solid var(--color-divider);overflow:hidden}
   #view-camera .stage img, #view-settings .stage img{
     width:100%;height:100%;object-fit:cover;display:block}
   #view-camera .hud{margin-top:0;display:flex;flex-direction:column;gap:18px;height:100%;justify-content:center}
   #view-camera .hcard{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;
                       padding:18px 24px;border-radius:24px;
                       border:1px solid var(--color-divider);box-shadow:var(--shadow-sm)}
   #view-camera .hcard.wide{flex:1.35;justify-content:flex-start}
   #view-camera .hcard .k{font-size:11.5px;letter-spacing:.1em}
   #view-camera .hcard .v{font-size:24px;margin-top:4px}
   #view-camera .hcard.wide .v{font-size:16px;line-height:1.55}

   /* Settings Right Panel on Desktop: Matches stage height seamlessly */
   .settings-panel{display:flex;flex-direction:column;height:100%;justify-content:center}
   #settings-card{display:flex;flex-direction:column;gap:var(--space-2);padding:18px 24px;
                  border-radius:24px;border:1px solid var(--color-divider);box-shadow:var(--shadow-sm);background:var(--color-surface)}
   .settings-header{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px}
   .settings-header h2{font-size:20px}
   .settings-header .hint{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 50%,transparent)}
   .settings-side{display:flex;flex-direction:column;gap:6px;background:var(--color-bg);
                  padding:8px 12px;border-radius:var(--radius-md);border:1px solid var(--color-divider);margin-top:4px}
   .settings-side .pills{display:flex;gap:6px;margin-top:0}
   .settings-side .pill{min-height:36px;padding:4px 12px;font-size:13.5px}

   /* Gallery Tab Desktop Layout (flows naturally from top) */
   #view-gallery{flex:none;width:100%;padding-top:var(--space-3)}
   #gal-grid .grid{grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:var(--space-4);width:100%}
   #gal-detail{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(360px,1fr);
               gap:32px;align-items:start}
   #gal-detail .back{grid-column:1 / -1;margin-bottom:var(--space-2);width:fit-content}
   #gal-detail .shot{max-height:72vh;object-fit:contain}
   #gal-detail .dmeta{margin-top:0}

   /* Floating Controls: Elevated & Enlarged Shutter on Top of Tabs */
    .dock{position:fixed;left:50%;transform:translateX(-50%);bottom:20px;z-index:10;
          display:flex;flex-direction:column;align-items:center;justify-content:center;
          gap:14px;padding:0;border-radius:0;
          background:transparent;backdrop-filter:none;-webkit-backdrop-filter:none;
          border:none;box-shadow:none;width:auto;max-width:none}
    .dock > *{width:auto;max-width:none}
    .toast{position:absolute;top:-36px;left:50%;transform:translateX(-50%);
           white-space:nowrap;background:var(--color-surface);padding:4px 14px;
           border-radius:999px;border:1px solid var(--color-divider);box-shadow:var(--shadow-sm);
           pointer-events:none}
    .toast:empty{display:none}
    .shutter{width:76px;height:76px;border-width:5px;margin:0 0 2px;box-shadow:var(--shadow-lg)}
    .shutter::after{width:52px;height:52px}
    .tabs{padding:5px;gap:6px;border:1px solid var(--color-divider);background:var(--color-surface);
          border-radius:999px;box-shadow:var(--shadow-md)}
    .tabs .pill{min-height:40px;padding:0 22px;font-size:14.5px}
    .pill:hover:not([aria-selected="true"]){border-color:var(--color-accent)}
  }

  @media (min-width: 1200px){
    body{max-width:1560px;width:95vw}
    #view-camera, #view-settings{grid-template-columns:minmax(0,1.75fr) minmax(380px,1fr);gap:40px}
    #view-camera .hcard .v{font-size:26px}
  }

  .hcard-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:2px}
  .rescan-btn-inline{display:none;align-items:center;gap:6px;background:var(--color-surface);
                     border:1.5px solid var(--color-divider);border-radius:999px;
                     padding:5px 16px;font-size:13.5px;font-weight:600;font-family:var(--font-heading);
                     color:var(--color-text);cursor:pointer;box-shadow:var(--shadow-sm);
                     transition:background .15s ease,border-color .15s ease,transform .12s ease,box-shadow .15s ease}
  .rescan-btn-inline:hover{border-color:var(--color-accent);background:var(--color-accent-2-100);box-shadow:var(--shadow-md)}
  .rescan-btn-inline:active{transform:scale(.95)}
  .rescan-btn-mobile{grid-column:1/-1;min-height:44px;font-size:14px;border:1px solid var(--color-divider);
                     background:var(--color-surface);color:var(--color-text);cursor:pointer;
                     display:inline-flex;align-items:center;justify-content:center;gap:6px}

  @media (min-width: 900px){
    .rescan-btn-mobile{display:none !important}
    .rescan-btn-inline{display:inline-flex}
  }

  @media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
</style></head><body class="cam">

<header class="topbar">
  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#036819"
       stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
    <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
  </svg>
  <span class="brand">Plant Health</span>
  <span class="conn" id="conn">connecting</span>
</header>

<main id="view-camera">
  <div class="stage">
    <img id="live" alt="Live camera view from the kiosk">
    <div class="live"><i></i>LIVE</div>
    <div class="veil" id="veil"><b>No live frames</b><span id="veiltext">the kiosk stream went quiet</span></div>
  </div>
  <div class="hud">
    <div class="hcard" id="h-type-card">
      <div class="hcard-top">
        <div class="k">Plant type</div>
        <button class="rescan-btn-inline" id="btn-rescan-inline" type="button" title="Rescan Plant">↻ Retry</button>
      </div>
      <div class="v" id="h-crop">—</div>
    </div>
    <div class="hcard" id="h-health-card">
      <div class="k">Plant health</div><div class="v" id="h-health">—</div>
      <span class="conf" id="h-conf" hidden></span>
    </div>
    <div class="hcard wide" id="h-notes-card">
      <div class="k">Notes</div><div class="v" id="h-notes">Looking for plant</div>
    </div>
    <button class="pill rescan-btn-mobile" id="btn-rescan" type="button">↻ Retry Scan</button>
  </div>
</main>

<main id="view-gallery" hidden>
  <div id="gal-grid">
    <div class="sect"><h2>Gallery</h2><span class="hint" id="count"></span></div>
    <div class="grid" id="gallery"></div>
  </div>
  <div id="gal-detail" hidden>
    <button class="back" id="back" type="button">&#8592; All scans</button>
    <img class="shot" id="shot" alt="Saved scan">
    <div class="dmeta">
      <div class="name" id="d-name"></div>
      <div class="when" id="d-when"></div>
      <div id="d-tags"></div>
    </div>
    <div class="dnotes" id="d-notes"></div>
    <div class="acts">
      <a id="d-dl" download>Download</a>
      <button id="d-del" type="button">Delete</button>
    </div>
  </div>
</main>

<main id="view-settings" hidden>
  <div class="stage">
    <img id="live-settings" alt="Live camera feed for color calibration">
    <div class="live"><i></i>LIVE</div>
  </div>
  <div class="settings-panel">
    <div class="card" id="settings-card">
      <div class="settings-header">
        <h2>Colour Profile</h2>
        <span class="hint">commits on release</span>
      </div>
      <div id="sliders"></div>
      <div class="settings-side">
        <div class="pills">
          <button class="pill" data-prof="night">Night</button>
          <button class="pill" data-prof="morning">Morning</button>
          <button class="pill" id="reset">Reset</button>
        </div>
        <div class="pills">
          <button class="pill sub" data-save="night">Save as Night</button>
          <button class="pill sub" data-save="morning">Save as Morning</button>
        </div>
      </div>
    </div>
  </div>
</main>

<div class="dock">
  <div class="toast" id="toast"></div>
  <button class="shutter" id="snap" aria-label="Capture a scan" title="Capture scan (Space/Enter)"></button>
  <div class="tabs" role="tablist" aria-label="Views">
    <button class="pill" role="tab" data-tab="camera" aria-selected="true">Camera</button>
    <button class="pill" role="tab" data-tab="gallery" aria-selected="false">Gallery</button>
    <button class="pill" role="tab" data-tab="settings" aria-selected="false">Settings</button>
  </div>
</div>

<script>
const TOKEN = "__TOKEN__";
const q = s => document.querySelector(s);
const url = p => p + (p.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN);
let toastTimer = null;
function toast(text, bad) {
  const el = q("#toast");
  el.textContent = text || "";
  el.classList.toggle("bad", !!bad);
  clearTimeout(toastTimer);
  if (text) toastTimer = setTimeout(() => { el.textContent = ""; el.classList.remove("bad"); }, 4000);
}

async function api(path, opts) {
  try {
    const r = await fetch(url(path), opts || {});
    return await r.json();
  } catch (e) {
    return {ok: false, error: "no answer from the kiosk"};
  }
}

// — live view —
const live = q("#live");
const liveSettings = q("#live-settings");
let streaming = false;
function startStream() {
  streaming = true;
  const s = url("/stream.mjpg") + "&t=" + Date.now();
  if (live) live.src = s;
  if (liveSettings) liveSettings.src = s;
}
function freezeStream() {
  if (!streaming) return;
  streaming = false;
  let still = "";
  try {
    const srcImg = (tab === "settings" && liveSettings) ? liveSettings : live;
    if (srcImg && srcImg.naturalWidth) {
      const c = document.createElement("canvas");
      c.width = srcImg.naturalWidth; c.height = srcImg.naturalHeight;
      c.getContext("2d").drawImage(srcImg, 0, 0);
      still = c.toDataURL("image/jpeg", 0.7);
    }
  } catch (e) { still = ""; }
  if (still) {
    if (live) live.src = still;
    if (liveSettings) liveSettings.src = still;
  }
}
live.addEventListener("error", () => {
  if (streaming) setStale(true, "the stream connection dropped");
});

let staleNow = false;
function setStale(on, why) {
  staleNow = !!on;
  document.body.classList.toggle("stale", staleNow);
  q("#veiltext").textContent = why || "the kiosk stream went quiet";
  const conn = q("#conn");
  conn.textContent = staleNow ? "no signal" : "streaming";
  conn.classList.toggle("bad", staleNow);
  conn.classList.remove("idle");
}

// — tabs —
const VIEWS = ["camera", "gallery", "settings"];
let tab = "camera";
let galleryDirty = true;
let colorLoaded = false;

function showTab(name) {
  if (!VIEWS.includes(name)) return;
  tab = name;
  for (const v of VIEWS) q("#view-" + v).hidden = (v !== name);
  document.querySelectorAll("[data-tab]").forEach(b =>
    b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.body.classList.toggle("cam", name === "camera");
  q("#snap").hidden = (name !== "camera");
  if (name === "camera" || name === "settings") {
    startStream();
    const conn = q("#conn");
    conn.textContent = "streaming";
    conn.classList.remove("idle", "bad");
    if (name === "camera") startPolling();
    else stopPolling();
  } else {
    stopPolling();
    freezeStream();
    const conn = q("#conn");
    conn.textContent = "paused";
    conn.classList.remove("bad");
    conn.classList.add("idle");
  }
  if (name === "gallery") { showGrid(); if (galleryDirty) loadGallery(); }
  if (name === "settings" && !colorLoaded) {
    colorLoaded = true;
    api("/api/color").then(paintColor);
  }
  window.scrollTo(0, 0);
}
document.querySelectorAll("[data-tab]").forEach(b => b.onclick = () => showTab(b.dataset.tab));

// — shutter —
q("#snap").onclick = async () => {
  const b = q("#snap");
  b.disabled = true;
  toast("capturing...");
  const r = await api("/shutter", {method: "POST"});
  b.disabled = false;
  toast(r.ok ? "saved on the kiosk" : ("not captured - " + (r.reason || r.error || "busy")), !r.ok);
  // The kiosk grades and writes the row after the shutter returns, and how
  // long that takes varies, so the Gallery tab is simply marked stale and
  // refetches the next time the user looks at it.
  if (r.ok) galleryDirty = true;
};

// — colour —
// Dragging never touches the network: `input` fires continuously and only
// repaints the number, `change` fires once when the finger lifts and is the
// only thing that POSTs. A wheel over a range input silently changes it in
// Chrome, which has altered live camera settings by accident before, so wheel
// is swallowed.
const sliders = {};
function fmt(v) { return (+v).toFixed(2); }

function buildSliders(state) {
  const host = q("#sliders");
  if (host.childElementCount) return;
  for (const s of (state.sliders || [])) {
    const row = document.createElement("div"); row.className = "row";
    const lab = document.createElement("label"); lab.textContent = s.label;
    const out = document.createElement("output");
    const inp = document.createElement("input"); inp.type = "range";
    const rng = (state.ranges || {})[s.name] || [0, 1];
    inp.min = rng[0]; inp.max = rng[1]; inp.step = 0.01;
    inp.id = "sl-" + s.name; lab.htmlFor = inp.id;
    inp.setAttribute("aria-label", s.label);
    inp.addEventListener("input", () => {
      inp.dataset.drag = "1";
      out.textContent = fmt(inp.value);
      out.className = "pending";
    });
    inp.addEventListener("change", () => { inp.dataset.drag = ""; commit(s.name, inp.value); });
    inp.addEventListener("wheel", e => { e.preventDefault(); }, {passive: false});
    row.append(lab, out, inp); host.append(row);
    sliders[s.name] = {inp, out, label: s.label};
  }
}

async function commit(name, value) {
  const el = sliders[name];
  // A wheel over a range still emits `change` in Chrome even with the wheel
  // default prevented, and a tap that does not move the thumb emits one too.
  // Neither is a real edit, so nothing goes out unless the value actually moved.
  if (el.confirmed !== undefined && Math.abs(+value - el.confirmed) < 0.005) {
    el.out.className = "";
    return;
  }
  el.out.className = "pending";
  const r = await api("/api/color", {method: "POST",
    body: JSON.stringify({action: "set", name: name, value: +value})});
  if (!r || !r.ok) {
    el.out.className = "";
    toast((r && r.error) || "colour rejected", true);
    return;
  }
  paintColor(r);
  el.out.className = "saved";
  toast(el.label + " set to " + fmt((r.profile || {})[name]));
  setTimeout(() => { if (el.out.className === "saved") el.out.className = ""; }, 1200);
}

function paintColor(state) {
  if (!state || !state.profile) return;
  buildSliders(state);
  for (const [name, el] of Object.entries(sliders)) {
    const v = state.profile[name];
    if (v === undefined) continue;
    el.confirmed = +v;
    if (el.inp.dataset.drag === "1") continue;
    el.inp.value = v;
    el.out.textContent = fmt(v);
  }
  document.querySelectorAll("[data-prof]").forEach(b =>
    b.classList.toggle("on", b.dataset.prof === state.active));
}

async function post(body, said) {
  const r = await api("/api/color", {method: "POST", body: JSON.stringify(body)});
  if (!r || !r.ok) { toast((r && r.error) || "not applied", true); return; }
  for (const el of Object.values(sliders)) el.out.className = "";
  paintColor(r);
  toast(said);
}
document.querySelectorAll("[data-prof]").forEach(b =>
  b.onclick = () => post({action: "activate", name: b.dataset.prof}, b.textContent + " profile active"));
document.querySelectorAll("[data-save]").forEach(b =>
  b.onclick = () => post({action: "save", name: b.dataset.save}, "saved as " + b.dataset.save));
q("#reset").onclick = () => post({action: "reset"}, "colour reset");

// — gallery —
const GRADES = ["healthy", "mild", "critical", "dead"];
function gradeOf(s) { return String((s && s.health) || "").toLowerCase(); }
function whenOf(s) { return String((s && s.created_at) || "").replace("T", " ").replace("Z", ""); }
function nameOf(s) { return (s && (s.crop || s.named_plant)) || "Unidentified"; }
function makeTag(text, grade) {
  const tag = document.createElement("span");
  tag.className = "tag" + (GRADES.includes(grade) ? " " + grade : "");
  tag.textContent = text;
  return tag;
}

function showGrid() { q("#gal-grid").hidden = false; q("#gal-detail").hidden = true; }
q("#back").onclick = () => { showGrid(); window.scrollTo(0, 0); };

let openScan = null;
function openDetail(s) {
  openScan = s;
  q("#shot").src = url("/photo?id=" + s.id);
  q("#shot").alt = "Saved scan of " + nameOf(s);
  q("#d-name").textContent = nameOf(s);
  q("#d-when").textContent = whenOf(s);
  const tags = q("#d-tags"); tags.textContent = "";
  const grade = gradeOf(s);
  tags.append(makeTag(grade || "ungraded", grade));
  if (s.confidence !== null && s.confidence !== undefined) {
    const c = makeTag(Math.round(s.confidence) + "% confident", "");
    c.style.marginLeft = "6px";
    tags.append(c);
  }
  q("#d-notes").textContent = s.tip || "No notes were saved with this scan.";
  q("#d-dl").href = url("/photo?id=" + s.id + "&download=1");
  q("#d-dl").onclick = async (e) => {
    e.preventDefault();
    if (!openScan) return;
    try {
      toast("downloading photo…");
      const res = await fetch(url("/photo?id=" + openScan.id + "&download=1"));
      if (!res.ok) throw new Error("fetch failed");
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      const ext = blob.type === "image/png" ? ".png" : ".jpg";
      const plant = (openScan.crop || openScan.named_plant || "plant").toLowerCase().replace(/[^a-z0-9]/g, "_");
      a.download = "plant_scan_" + openScan.id + "_" + plant + ext;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
      toast("photo downloaded");
    } catch (err) {
      toast("download failed: " + err.message, true);
    }
  };
  const del = q("#d-del");
  del.textContent = "Delete"; del.classList.remove("armed"); del.dataset.armed = "";
  q("#gal-grid").hidden = true;
  q("#gal-detail").hidden = false;
  window.scrollTo(0, 0);
}

q("#d-del").onclick = async () => {
  const del = q("#d-del");
  if (!openScan) return;
  if (Date.now() > +(del.dataset.armed || 0)) {
    del.dataset.armed = String(Date.now() + 2500);
    del.textContent = "Tap again"; del.classList.add("armed");
    setTimeout(() => {
      if (Date.now() > +(del.dataset.armed || 0)) {
        del.textContent = "Delete"; del.classList.remove("armed");
      }
    }, 2600);
    return;
  }
  del.dataset.armed = "";
  const res = await api("/api/scan?id=" + openScan.id + "&confirm=yes", {method: "DELETE"});
  toast(res.ok ? "scan deleted" : (res.error || "delete failed"), !res.ok);
  del.textContent = "Delete"; del.classList.remove("armed");
  if (res.ok) { showGrid(); loadGallery(); }
};

async function loadGallery() {
  const r = await api("/api/scans");
  galleryDirty = false;
  const host = q("#gallery"); host.textContent = "";
  if (!r.ok || !r.scans || !r.scans.length) {
    const e = document.createElement("div"); e.className = "empty";
    e.textContent = r.ok ? "No scans yet. Tap the shutter on the Camera tab." : "Gallery unavailable.";
    host.append(e); q("#count").textContent = "";
    return;
  }
  q("#count").textContent = r.scans.length + (r.scans.length === 1 ? " scan" : " scans");
  for (const s of r.scans) {
    const el = document.createElement("button");
    el.className = "tile"; el.type = "button";
    el.setAttribute("aria-label", "Open scan of " + nameOf(s));
    const img = document.createElement("img");
    img.src = url("/thumb?id=" + s.id); img.loading = "lazy"; img.alt = "";
    const meta = document.createElement("div"); meta.className = "meta";
    const name = document.createElement("div"); name.className = "name";
    name.textContent = nameOf(s);
    const when = document.createElement("div"); when.className = "when";
    when.textContent = whenOf(s);
    const grade = gradeOf(s);
    meta.append(name, when, makeTag(grade || "ungraded", grade));
    el.append(img, meta);
    el.onclick = () => openDetail(s);
    host.append(el);
  }
}

// — status: liveness plus the HUD text —
//
// 1.2s. The kiosk re-grades every couple of seconds, so this is fast enough
// that the cards never visibly trail the picture, and /api/status is a flat
// JSON handler that reads six attributes without marshalling onto the Tk
// thread — under one request a second next to a 12 fps stream is noise. It
// also keeps the staleness rule (frame_age > 4s) well inside its own window.
const POLL_MS = 1200;
let pollTimer = null;
let misses = 0;

function paintHud(hud) {
  if (!hud) return;
  q("#h-crop").textContent = hud.crop || "—";
  q("#h-health").textContent = hud.health_label || "—";
  q("#h-health-card").className =
    "hcard" + (GRADES.includes(hud.health) ? " " + hud.health : "");
  const conf = q("#h-conf");
  if (hud.confidence === null || hud.confidence === undefined) {
    conf.hidden = true;
    conf.textContent = "";
  } else {
    conf.hidden = false;
    conf.textContent = Math.round(hud.confidence) + "% CONFIDENT";
  }
  const notes = q("#h-notes");
  notes.textContent = hud.notes || "—";
  if (hud.notes_extra) {
    const em = document.createElement("em");
    em.textContent = " " + hud.notes_extra;
    notes.append(em);
  }
  q("#h-notes-card").className = "hcard wide" + (hud.tone === "warn" ? " warn" : "");
}

async function poll() {
  const r = await api("/api/status");
  if (!r.ok) { misses += 1; setStale(true, "the kiosk is not answering"); return; }
  paintHud(r.hud);
  const bad = r.frame_age > 4;
  if (bad && !staleNow) misses += 1;
  const age = Math.round(r.frame_age);
  setStale(bad, (!r.seq || age > 3600)
    ? "the kiosk has not sent a frame yet"
    : ("no new frame for " + age + "s"));
  if (!bad) misses = 0;
  if (misses === 4) { misses = 0; startStream(); }   // one quiet retry of the stream
}

function startPolling() {
  if (pollTimer) return;
  poll();
  pollTimer = setInterval(poll, POLL_MS);
}
function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
  misses = 0;
}
// A backgrounded phone should not keep the Pi encoding for a screen nobody is
// looking at either.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { stopPolling(); freezeStream(); }
  else if (tab === "camera") { startStream(); startPolling(); }
});

document.addEventListener("keydown", (e) => {
  if (e.target && ["input", "textarea", "select"].includes(e.target.tagName.toLowerCase())) return;
  if ((e.code === "Space" || e.key === " " || e.key === "Enter") && tab === "camera") {
    const snap = q("#snap");
    if (snap && !snap.disabled && !snap.hidden) {
      e.preventDefault();
      snap.click();
    }
  }
});

q("#live").addEventListener("click", async (e) => {
  const rect = e.target.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  try {
    const res = await api("/api/tap", {
      method: "POST",
      body: JSON.stringify({ x: nx, y: ny })
    });
    if (res.ok) {
      poll();
    }
  } catch (err) {}
});

const triggerRescan = async () => {
  toast("Rescanning plant…");
  try {
    const res = await api("/api/retry", { method: "POST" });
    if (res.ok) poll();
  } catch (e) {}
};
const btnRescan = q("#btn-rescan");
if (btnRescan) btnRescan.onclick = triggerRescan;
const btnRescanInline = q("#btn-rescan-inline");
if (btnRescanInline) btnRescanInline.onclick = triggerRescan;

showTab("camera");
</script></body></html>
"""
)


for _name, _value in (
    ("__CREAM__", CREAM),
    ("__SURFACE__", SURFACE),
    ("__TEXT__", TEXT),
    ("__ACCENT__", ACCENT),
    ("__ALERT__", ALERT),
):
    LOGIN_HTML = LOGIN_HTML.replace(_name, _value)
    PAGE_HTML = PAGE_HTML.replace(_name, _value)
