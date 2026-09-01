"""Capture indicator LED on a Raspberry Pi GPIO pin.

A plain LED (BCM 17 by default) lights for a few seconds every time a photo is
captured, so an operator standing away from the touchscreen can see that the
shutter really fired.

Everything here degrades to a silent no-op when gpiozero is missing or the pin
cannot be claimed: the kiosk must never fail because of an indicator light.
"""

from __future__ import annotations

import threading

DEFAULT_PIN = 17
DEFAULT_HOLD = 3.0


class CaptureLight:
    """Blink an indicator LED for `hold` seconds on each `pulse()`."""

    def __init__(
        self,
        pin: int = DEFAULT_PIN,
        *,
        active_high: bool = True,
        hold: float = DEFAULT_HOLD,
        enabled: bool = True,
    ) -> None:
        self.pin = int(pin)
        self.active_high = bool(active_high)
        self.hold = max(0.05, float(hold))
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._gen = 0
        self._led = None
        self._closed = False
        self._backend = "disabled"
        self._reason = ""
        if not enabled:
            self._reason = "capture light disabled by flag"
            return
        self._open()

    # — setup —

    def _open(self) -> None:
        try:
            import gpiozero
        except Exception as exc:  # ImportError, or a broken install
            self._backend = "none"
            self._reason = f"gpiozero unavailable ({exc.__class__.__name__}: {exc})"
            print(f"[capture-light] off: {self._reason}")
            return
        try:
            self._led = gpiozero.LED(self.pin, active_high=self.active_high, initial_value=False)
            self._backend = "gpiozero (default pin factory)"
            return
        except Exception as exc:
            first = f"{exc.__class__.__name__}: {exc}"
        # Default factory failed — try lgpio explicitly (Debian trixie / Pi 5 path).
        try:
            from gpiozero.pins.lgpio import LGPIOFactory

            self._led = gpiozero.LED(
                self.pin,
                active_high=self.active_high,
                initial_value=False,
                pin_factory=LGPIOFactory(),
            )
            self._backend = "gpiozero (LGPIOFactory)"
            return
        except Exception as exc:
            second = f"{exc.__class__.__name__}: {exc}"
        self._led = None
        self._backend = "none"
        self._reason = f"BCM {self.pin} unavailable (default: {first}; lgpio: {second})"
        print(f"[capture-light] off: {self._reason}")

    # — state —

    @property
    def active(self) -> bool:
        """True when a real GPIO pin is being driven."""
        return self._led is not None and not self._closed

    @property
    def backend(self) -> str:
        """Short human-readable description of the resolved backend."""
        if not self.active:
            return f"off - {self._reason}" if self._reason else "off"
        return f"{self._backend}, BCM {self.pin}, active-{'high' if self.active_high else 'low'}"

    # — use —

    def pulse(self) -> None:
        """Light the LED now and schedule it off after `hold` seconds."""
        with self._lock:
            if self._led is None or self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            # cancel() loses the race when the timer has already fired and is
            # blocked on this lock: that stale _expire would switch the LED off
            # moments after this pulse turned it on. The generation token makes
            # a late expiry a no-op.
            self._gen += 1
            gen = self._gen
            try:
                self._led.on()
            except Exception:
                return
            timer = threading.Timer(self.hold, self._expire, args=(gen,))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _expire(self, gen: int) -> None:
        with self._lock:
            if gen != self._gen:
                return
            self._timer = None
            if self._led is None or self._closed:
                return
            try:
                self._led.off()
            except Exception:
                pass

    def close(self) -> None:
        """Cancel any pending timer, turn the LED off and release the pin."""
        with self._lock:
            self._closed = True
            self._gen += 1
            if self._timer is not None:
                try:
                    self._timer.cancel()
                except Exception:
                    pass
                self._timer = None
            led, self._led = self._led, None
        if led is None:
            return
        for step in (led.off, led.close):
            try:
                step()
            except Exception:
                pass


def _self_test(argv: list[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Capture indicator LED self-test")
    parser.add_argument("--pin", type=int, default=DEFAULT_PIN)
    parser.add_argument("--active-low", action="store_true")
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)

    light = CaptureLight(
        args.pin,
        active_high=not args.active_low,
        hold=args.hold,
        enabled=True,
    )
    print(f"[capture-light] backend: {light.backend}")
    if not light.active:
        print("[capture-light] no GPIO - nothing to blink.")
        print("  Checks: running on the Pi? user in the 'gpio' group? /dev/gpiochip* present?")
        print("  Wiring: BCM 17 = physical pin 11, resistor to LED anode, cathode to pin 9 (GND).")
        light.close()
        return 1
    print(f"[capture-light] blinking {args.count}x on BCM {args.pin}, {args.hold:.2f}s each")
    try:
        for i in range(1, max(1, args.count) + 1):
            print(f"  pulse {i}/{args.count} - LED should be ON")
            light.pulse()
            time.sleep(args.hold + 0.4)
    except KeyboardInterrupt:
        print("[capture-light] interrupted")
    finally:
        light.close()
    print("[capture-light] done - LED off, pin released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
