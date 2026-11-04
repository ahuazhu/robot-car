"""Package the normal Python application and emit a self-contained installer."""

import argparse
import base64
import io
from pathlib import Path
import zipfile

APP_FILES = ("main.py", "app.yaml", "calibration.json", "runtime.json", "curve_fit_kernel.c")


def build_archive(source_root):
    """Preserve module namespaces and package resources; never rewrite imports."""
    source_root = Path(source_root)
    paths = [source_root / name for name in APP_FILES]
    package = source_root / "robot"
    for required in ("__init__.py", "application.py", "interface/preview.html"):
        if not (package / required).is_file():
            raise ValueError("Missing robot package file: " + required)
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Package symlinks are not deployable: " + str(relative))
        if path.is_file() and path.suffix not in (".pyc", ".pyo", ".so", ".dylib", ".o"):
            paths.append(path)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError("Application symlinks are not deployable: " + str(path))
            data = path.read_bytes()
            name = path.relative_to(source_root).as_posix()
            if path.suffix == ".py":
                compile(data, name, "exec")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return payload.getvalue()


def build(source_root, output_dir, *, drive=False):
    source_root, output_dir = Path(source_root), Path(output_dir)
    payload = build_archive(source_root)
    template = (source_root / "tools" / "install_app.py").read_text()
    compile(template, "tools/install_app.py", "exec")
    encoded = base64.b64encode(payload).decode("ascii")
    installer = (
        template
        + '\n\nif __name__ == "__main__":\n    import base64\n    install_app(base64.b64decode('
        + repr(encoded)
        + "), force_preview="
        + repr(not drive)
        + ")\n"
    )
    compile(installer, "install_curve_follow.py", "exec")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "build"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / "curve_follow.zip"
    archive_path.write_bytes(payload)
    name = "install_curve_follow_drive.py" if drive else "install_curve_follow.py"
    installer_path = output_dir / name
    installer_path.write_text(installer)
    return installer_path, archive_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Allow configured driving; deployment still starts unarmed",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Local artifact directory (default: project .maixpy)"
    )
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parent
    output = args.output_dir or source_root.parents[1] / ".maixpy"
    installer, archive = build(source_root, output, drive=args.drive)
    print("Built package:", archive)
    print(
        "Installer:",
        installer,
        "(configured driving, unarmed)" if args.drive else "(forced preview)",
    )


if __name__ == "__main__":
    main()
