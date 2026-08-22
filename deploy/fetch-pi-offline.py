import gzip
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ROOT / "vendor" / "wheels"
DEBS = ROOT / "vendor" / "debs"
CURL = ["curl.exe", "-fsSL", "--retry", "5", "-A", "Mozilla/5.0"]

PY_TAGS = (("311", "cp311"), ("312", "cp312"), ("313", "cp313"))
TORCH_PKGS = ["torch", "torchvision"]
TORCH_DEPS = [
    "typing-extensions",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "filelock",
    "mpmath",
    "markupsafe",
]
PYPI_PKGS = ["pillow", "pyyaml", "opencv-python>=4.8,<5", "numpy"]
PLATFORMS = [
    "linux_aarch64",
    "manylinux_2_35_aarch64",
    "manylinux_2_28_aarch64",
    "manylinux_2_17_aarch64",
    "manylinux2014_aarch64",
]
DEB_SEEDS = ("python3-tk", "python3-venv", "python3.11-tk", "python3.11-venv")
SKIP_DEB = {
    "libc6",
    "libgcc-s1",
    "libstdc++6",
    "zlib1g",
    "libssl3",
    "libbz2-1.0",
    "liblzma5",
    "libffi8",
    "libexpat1",
    "libncursesw6",
    "libtinfo6",
    "libreadline8",
    "libsqlite3-0",
    "libuuid1",
    "python3",
    "python3-minimal",
    "python3.11",
    "python3.11-minimal",
    "libpython3.11-stdlib",
    "libpython3.11-minimal",
    "libpython3-stdlib",
    "dpkg",
    "tar",
    "gzip",
    "perl-base",
    "debconf",
    "ca-certificates",
}
MIRRORS = (
    ("http://deb.debian.org/debian", "bookworm", "main"),
    ("http://archive.raspberrypi.com/debian", "bookworm", "main"),
)


def curl(url: str, dest: Path | None = None) -> bytes | None:
    if dest is None:
        return subprocess.check_output([*CURL, url], timeout=180)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("have", dest.name)
        return None
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.check_call([*CURL, "-o", str(tmp), url], timeout=600)
    tmp.replace(dest)
    return None


def parse_packages(raw: bytes) -> dict[str, dict[str, str]]:
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    out: dict[str, dict[str, str]] = {}
    for block in text.split("\n\n"):
        rec: dict[str, str] = {}
        key = ""
        for line in block.splitlines():
            if not line:
                continue
            if line.startswith(" ") and key:
                rec[key] += "\n" + line[1:]
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            rec[key] = val.strip()
        name = rec.get("Package")
        if name and name not in out:
            out[name] = rec
    return out


def dep_names(field: str) -> list[str]:
    names = []
    for part in field.split(","):
        first = part.split("|")[0].strip()
        name = re.split(r"\s*\(", first, maxsplit=1)[0].strip()
        if name and name not in ("Pre-Depends",):
            names.append(name)
    return names


def collect_debs() -> None:
    index: dict[str, dict[str, str]] = {}
    bases: dict[str, str] = {}
    for base, dist, comp in MIRRORS:
        url = f"{base}/dists/{dist}/{comp}/binary-arm64/Packages.gz"
        print("index", url)
        try:
            blob = curl(url)
        except subprocess.CalledProcessError:
            print("skip index", url)
            continue
        assert blob is not None
        parsed = parse_packages(blob)
        for name, rec in parsed.items():
            if name not in index:
                index[name] = rec
                bases[name] = base + "/"
    want = set(DEB_SEEDS)
    seen: set[str] = set()
    while True:
        extra = set()
        for name in list(want):
            rec = index.get(name)
            if not rec:
                continue
            for field in ("Depends", "Pre-Depends"):
                if field in rec:
                    extra.update(dep_names(rec[field]))
        extra -= SKIP_DEB
        extra -= want
        extra -= seen
        if not extra:
            break
        want |= extra
    DEBS.mkdir(parents=True, exist_ok=True)
    for name in sorted(want):
        if name in SKIP_DEB:
            continue
        rec = index.get(name)
        if not rec or "Filename" not in rec:
            print("missing package", name)
            continue
        url = urljoin(bases[name], rec["Filename"])
        dest = DEBS / Path(rec["Filename"]).name
        print("deb", dest.name)
        curl(url, dest)


def pip_download(py_ver: str, abi: str, pkgs: list[str], extra: list[str], platforms: list[str]) -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    plat = []
    for p in platforms:
        plat.extend(["--platform", p])
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pip",
        "download",
        "-d",
        str(WHEELS),
        "--python-version",
        py_ver,
        "--abi",
        abi,
        "--implementation",
        "cp",
        "--only-binary=:all:",
        *plat,
        *extra,
        *pkgs,
    ]
    print("download", abi, " ".join(pkgs), extra[:2])
    subprocess.check_call(cmd)


def collect_wheels() -> None:
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    for py_ver, abi in PY_TAGS:
        print("wheels", abi)
        try:
            pip_download(
                py_ver,
                abi,
                TORCH_PKGS,
                ["--index-url", "https://download.pytorch.org/whl/cpu", "--no-deps"],
                ["linux_aarch64"],
            )
        except subprocess.CalledProcessError:
            print("no torch wheels for", abi)
        try:
            pip_download(py_ver, abi, TORCH_DEPS + PYPI_PKGS, [], PLATFORMS)
        except subprocess.CalledProcessError:
            print("no pypi wheels for", abi)
    subprocess.check_call(
        [py, "-m", "pip", "download", "-d", str(WHEELS), "--only-binary=:all:", "pip", "setuptools", "wheel"]
    )


def main() -> None:
    collect_wheels()
    collect_debs()
    print("wheels", len(list(WHEELS.glob("*.whl"))), "debs", len(list(DEBS.glob("*.deb"))))


if __name__ == "__main__":
    sys.exit(main())
