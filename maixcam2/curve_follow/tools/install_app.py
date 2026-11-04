"""Device installer template. Build embeds a package; importing this file is inert.

Run the generated installer through the MaixPy helper, which stops the previous
application first. Only this application's managed files are replaced. A failed
registration or startup restores the saved files and leaves the car stopped.
"""

import configparser
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.request
import zipfile

APP_ID = "robot_curve_follow"
APP_ROOT = Path("/maixapp/apps") / APP_ID
MANAGED = (
    "robot",
    "main.py",
    "calibration.json",
    "runtime.json",
    "curve_fit_kernel.c",
    "curve_fit_kernel.so",
    "app.yaml",
)
NATIVE_SYMBOLS = (
    "curve_fit_local",
    "curve_agree_local",
    "curve_fit_chain",
    "curve_project_polyline",
    "curve_boundary_chains",
    "curve_select_pursuit",
    "curve_agree_paths",
    "curve_nearest_samples",
)
REQUIRED = {
    "main.py",
    "robot/__init__.py",
    "robot/application.py",
    "robot/interface/preview.html",
    "app.yaml",
    "calibration.json",
    "runtime.json",
    "curve_fit_kernel.c",
}


def unpack_bundle(payload, destination):
    """Validate the complete archive before writing any of its members."""
    destination = Path(destination)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                member.is_dir()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or path.as_posix() != member.filename
                or not path.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Invalid bundle path: " + member.filename)
            allowed = path.parts[0] == "robot" or member.filename in set(MANAGED) - {
                "robot",
                "curve_fit_kernel.so",
            }
            if not allowed or member.filename in names:
                raise ValueError("Unexpected or duplicate bundle member: " + member.filename)
            total += member.file_size
            if total > 20 * 1024 * 1024 or len(members) > 1000:
                raise ValueError("Application bundle is unexpectedly large")
            names.add(member.filename)
        if not REQUIRED <= names:
            raise ValueError("Missing application files: " + ", ".join(sorted(REQUIRED - names)))
        contents = {member.filename: archive.read(member) for member in members}
    for name, data in contents.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        if name in ("runtime.json", "calibration.json") and not isinstance(json.loads(data), dict):
            raise ValueError("Configuration must be a JSON object: " + name)
    for name, data in contents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def configure_install_mode(staging, *, force_preview):
    """Persist preview protection for desktop restarts without launch variables."""
    if force_preview:
        path = staging / "runtime.json"
        runtime = json.loads(path.read_text())
        runtime.update(allow_drive=False, preview_only=True)
        path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n")


def compile_native(staging):
    library_path = staging / "curve_fit_kernel.so"
    result = subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O3",
            "-fPIC",
            "-shared",
            str(staging / "curve_fit_kernel.c"),
            "-o",
            str(library_path),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("Native kernel compilation failed: " + result.stdout + result.stderr)
    library = ctypes.CDLL(str(library_path))
    for symbol in NATIVE_SYMBOLS:
        getattr(library, symbol)
    print("NATIVE_KERNEL_READY", flush=True)


def remove_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def restore_path(source, target):
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


class AppFiles:
    """Keep an untouched backup while replacing only the owned package/files."""

    def __init__(self, root, backup):
        self.root, self.backup = root, backup
        self.changes = []

    def publish(self, staging):
        for name in MANAGED:
            target, saved = self.root / name, self.backup / name
            if target.is_symlink():
                raise ValueError("Application destination is a symlink: " + str(target))
            change = dict(name=name, moved=False, installed=False)
            self.changes.append(change)
            if target.exists():
                target.replace(saved)
                change["moved"] = True
            (staging / name).replace(target)
            change["installed"] = True

    def rollback(self):
        for change in reversed(self.changes):
            target = self.root / change["name"]
            if change["installed"]:
                remove_path(target)
            if change["moved"]:
                restore_path(self.backup / change["name"], target)


def register_app(root):
    subprocess.run(
        ["/usr/bin/python3", str(root.parent / "gen_app_info.py"), str(root.parent)], check=True
    )
    registered = configparser.ConfigParser(interpolation=None)
    registered.read(root.parent / "app.info")
    if registered.get(APP_ID, "exec", fallback=None) != "main.py":
        raise RuntimeError("Launcher does not select the current application entry")
    print("LAUNCHER_REGISTERED", APP_ID, flush=True)


def assert_preview_port_available():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", 8000))
        except OSError as exc:
            raise RuntimeError(
                "Preview port 8000 is occupied; stop the previous app through the MaixPy helper first"
            ) from exc


def read_json(endpoint):
    with urllib.request.urlopen("http://127.0.0.1:8000/" + endpoint, timeout=1) as response:
        return json.load(response)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def launch_and_verify(root, *, force_preview=True, timeout=30.0):
    env = dict(
        os.environ,
        MAIXPY_DEBUG_IMAGES="0",
        TARGET_TEST_SECONDS="0",
        TARGET_PREVIEW_ONLY="1" if force_preview else "0",
        START_LIGHT_DISARMED="1",
    )
    log_path = Path("/tmp/robot_curve_follow.log")
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["/usr/bin/python3", "-u", str(root / "main.py")],
            cwd=str(root),
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )
    received = set()
    try:
        while time.monotonic() < started + timeout and process.poll() is None:
            try:
                health = read_json("vision")
                status = read_json("status")
            except (OSError, ValueError):
                time.sleep(0.25)
                continue
            if (
                status.get("active") is not False
                or status.get("left") != 0
                or status.get("right") != 0
                or status.get("readiness", {}).get("start_light", {}).get("armed", False)
            ):
                raise RuntimeError("Unexpected armed or nonzero output state after deployment")
            motor = health.get("motor")
            if motor is not None:
                ack = motor.get("ack", [])
                if motor.get("armed") or len(ack) < 4 or ack[1:4] != [0, 0, 0]:
                    raise RuntimeError(
                        "Unexpected armed or nonzero UNO acknowledgement after deployment"
                    )
            if health.get("frame", 0) > 0 and health.get("frame_received_at", 0) >= started:
                received.add(health["frame"])
            if len(received) >= 3:
                break
            time.sleep(0.25)
        if len(received) < 3 or process.poll() is not None:
            raise RuntimeError("App did not produce three fresh, stopped telemetry frames")
        print(
            "FIRST_FRAMES_READY",
            health.get("app_version"),
            sorted(received),
            "active",
            status["active"],
            flush=True,
        )
        print(
            "PID",
            process.pid,
            "running",
            True,
            "ENTRY_SHA256",
            hashlib.sha256((root / "main.py").read_bytes()).hexdigest(),
            flush=True,
        )
        return process
    except Exception:
        stop_process(process)
        print(log_path.read_text(errors="replace")[-3000:], flush=True)
        raise


def install_app(payload, *, force_preview=True, root=APP_ROOT):
    root = Path(root)
    assert_preview_port_available()
    root.mkdir(parents=True, exist_ok=True)
    backup = (
        root
        / "backups"
        / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + str(time.time_ns()))
    )
    registry = root.parent / "app.info"
    previous_registry = registry.read_bytes() if registry.exists() else None
    with tempfile.TemporaryDirectory(prefix=".install-", dir=str(root)) as work:
        staging = Path(work)
        unpack_bundle(payload, staging)
        configure_install_mode(staging, force_preview=force_preview)
        compile_native(staging)
        backup.mkdir(parents=True)
        if previous_registry is not None:
            (backup / "app.info").write_bytes(previous_registry)
        files = AppFiles(root, backup)
        registry_touched = False
        try:
            files.publish(staging)
            registry_touched = True
            register_app(root)
            process = launch_and_verify(root, force_preview=force_preview)
        except Exception:
            files.rollback()
            if registry_touched:
                if previous_registry is None:
                    registry.unlink(missing_ok=True)
                else:
                    registry.write_bytes(previous_registry)
            print("INSTALL_FAILED_RESTORED", str(backup), flush=True)
            raise
    print(
        "DEVICE_BACKUP",
        str(backup),
        "BUNDLE_SHA256",
        hashlib.sha256(payload).hexdigest(),
        flush=True,
    )
    return process
