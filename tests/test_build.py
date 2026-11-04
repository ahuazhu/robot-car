"""Offline package/deployment tests; no camera, UART or device connection."""

import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from tests.support import APP_ROOT as ROOT
from maixcam2.curve_follow import build as builder
from maixcam2.curve_follow.tools import install_app as installer


def minimal_bundle(extra=None):
    files = {name: b"" for name in installer.REQUIRED}
    files.update(
        {
            "runtime.json": b"{}",
            "calibration.json": b"{}",
            "app.yaml": b"id: robot_curve_follow\n",
            "main.py": b"from robot.application import main\n",
            "robot/application.py": b"def main(app_dir):\n    pass\n",
            "robot/interface/preview.html": b"<html>preview</html>",
        }
    )
    files.update(extra or {})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


class BuildTests(unittest.TestCase):
    def test_package_keeps_imports_resources_and_excludes_generated_code(self):
        payload = builder.build_archive(ROOT)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertTrue(installer.REQUIRED <= set(archive.namelist()))
            self.assertEqual(archive.read("main.py"), (ROOT / "main.py").read_bytes())
            for name in archive.namelist():
                self.assertNotIn("__pycache__", name)
                self.assertNotIn("standalone", name)
                if name.endswith(".py"):
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            self.assertIn("robot/interface/preview.html", archive.namelist())
        self.assertEqual(payload, builder.build_archive(ROOT))

    def test_installers_are_import_safe_and_drive_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            for drive in (False, True):
                path, archive = builder.build(ROOT, folder, drive=drive)
                source = path.read_text()
                tree = ast.parse(source)
                namespace = {"__name__": "offline_installer_check"}
                exec(compile(tree, str(path), "exec"), namespace)
                self.assertTrue(callable(namespace["install_app"]))
                # Execute the real entry guard with installation intercepted;
                # its statement order and imports are not part of the contract.
                install = Mock()
                namespace.update(__name__="__main__", install_app=install)
                entry = ast.Module(body=[tree.body[-1]], type_ignores=[])
                exec(compile(entry, str(path), "exec"), namespace)
                install.assert_called_once_with(archive.read_bytes(), force_preview=not drive)

    def test_unpack_rejects_unsafe_paths_and_invalid_code_before_writes(self):
        for extra in (
            {"../outside.py": b""},
            {"robot/../../outside.py": b""},
            {"robot\\outside.py": b""},
            {"robot/broken.py": b"if invalid"},
            {"calibration.json": b"[]"},
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises((ValueError, SyntaxError)):
                    installer.unpack_bundle(minimal_bundle(extra), Path(folder))
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_native_compilation_checks_all_eight_required_symbols(self):
        library = Mock()
        with (
            patch.object(installer.subprocess, "run", return_value=Mock(returncode=0)),
            patch.object(installer.ctypes, "CDLL", return_value=library),
        ):
            installer.compile_native(Path("/offline/staging"))
        self.assertEqual(len(installer.NATIVE_SYMBOLS), 8)
        for name in installer.NATIVE_SYMBOLS:
            self.assertIn(name, library._mock_children)

    def test_registration_keeps_unrelated_legacy_app_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / installer.APP_ID
            root.mkdir()
            (root.parent / "app.info").write_text(
                "[robot_curve_follow]\nexec=main.py\n[robot_lane_follow]\nexec=main.py\n"
            )
            with patch.object(installer.subprocess, "run") as run:
                installer.register_app(root)
            run.assert_called_once()
            self.assertIn("[robot_lane_follow]", (root.parent / "app.info").read_text())

    def test_failure_restores_previous_package_registry_and_retains_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / installer.APP_ID
            (root / "robot").mkdir(parents=True)
            (root / "robot" / "old.py").write_text("old = True\n")
            (root / "main.py").write_text("old entry\n")
            (root / "keep.txt").write_text("unmanaged")
            registry = root.parent / "app.info"
            registry.write_text("old registry")

            def compile_fake(staging):
                (staging / "curve_fit_kernel.so").write_bytes(b"test kernel")

            def register_fake(app_root):
                registry.write_text("new registry")

            with (
                patch.object(installer, "assert_preview_port_available"),
                patch.object(installer, "compile_native", side_effect=compile_fake),
                patch.object(installer, "register_app", side_effect=register_fake),
                patch.object(
                    installer, "launch_and_verify", side_effect=RuntimeError("health failed")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "health failed"):
                    installer.install_app(minimal_bundle(), root=root)
            self.assertEqual((root / "main.py").read_text(), "old entry\n")
            self.assertTrue((root / "robot" / "old.py").is_file())
            self.assertFalse((root / "robot" / "application.py").exists())
            self.assertFalse((root / "curve_fit_kernel.so").exists())
            self.assertEqual(registry.read_text(), "old registry")
            self.assertEqual((root / "keep.txt").read_text(), "unmanaged")
            backups = list((root / "backups").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "main.py").read_text(), "old entry\n")

    def test_preview_and_drive_start_unarmed_and_verify_three_fresh_zero_frames(self):
        for preview in (False, True):
            with self.subTest(preview=preview), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "main.py").write_text("")
                process = Mock(pid=123)
                process.poll.return_value = None
                health = [
                    dict(
                        frame=i,
                        frame_received_at=100.0,
                        app_version="test",
                        motor=dict(armed=False, ack=[i, 0, 0, 0, 0]),
                    )
                    for i in (1, 2, 3)
                ]
                status = dict(
                    active=False, left=0, right=0, readiness=dict(start_light=dict(armed=False))
                )
                responses = [row for h in health for row in (h, status)]
                with (
                    patch.object(installer.subprocess, "Popen", return_value=process) as launch,
                    patch.object(installer.time, "monotonic", return_value=100.0),
                    patch.object(installer.time, "sleep"),
                    patch.object(installer, "read_json", side_effect=responses) as read,
                    patch.object(installer.Path, "open", unittest.mock.mock_open(read_data=b"")),
                ):
                    actual = installer.launch_and_verify(root, force_preview=preview)
                self.assertIs(actual, process)
                self.assertEqual(read.call_count, 6)
                env = launch.call_args.kwargs["env"]
                self.assertEqual(env["TARGET_PREVIEW_ONLY"], "1" if preview else "0")
                self.assertEqual(env["START_LIGHT_DISARMED"], "1")
                process.terminate.assert_not_called()

    def install_offline(self, root, runtime, *, preview, calibrated=True):
        calibration = json.loads((ROOT / "calibration.json").read_text())
        calibration.update(geometry_confirmed=True, camera_pose_verified=calibrated)
        payload = minimal_bundle(
            {
                "runtime.json": json.dumps(runtime).encode(),
                "calibration.json": json.dumps(calibration).encode(),
            }
        )

        def compile_fake(staging):
            (staging / "curve_fit_kernel.so").write_bytes(b"offline native kernel")

        with (
            patch.object(installer, "assert_preview_port_available"),
            patch.object(installer, "compile_native", side_effect=compile_fake),
            patch.object(installer, "register_app"),
            patch.object(installer, "launch_and_verify") as launch,
        ):
            installer.install_app(payload, root=root, force_preview=preview)
        launch.assert_called_once_with(root, force_preview=preview)
        # Desktop launch supplies none of the installer's environment overrides.
        from robot.config import RuntimeSettings

        reopened = RuntimeSettings.load(root, environ={})
        return reopened

    def test_forced_preview_remains_disabled_after_desktop_restart(self):
        runtime = json.loads((ROOT / "runtime.json").read_text())
        runtime.update(allow_drive=True, preview_only=False)
        original = dict(runtime)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / installer.APP_ID
            reopened = self.install_offline(root, runtime, preview=True)
            persisted = json.loads((root / "runtime.json").read_text())
            self.assertEqual(persisted, dict(original, allow_drive=False, preview_only=True))
            self.assertTrue(reopened.preview_only)
            self.assertEqual(runtime, original)

    def test_drive_install_retains_original_runtime_and_calibration_guards(self):
        for allow, preview_only, calibrated, expected in (
            (True, False, True, False),
            (False, False, True, True),
            (True, True, True, True),
            (True, False, False, True),
        ):
            with (
                self.subTest(allow=allow, preview=preview_only, calibrated=calibrated),
                tempfile.TemporaryDirectory() as folder,
            ):
                root = Path(folder) / installer.APP_ID
                runtime = json.loads((ROOT / "runtime.json").read_text())
                runtime.update(allow_drive=allow, preview_only=preview_only)
                reopened = self.install_offline(root, runtime, preview=False, calibrated=calibrated)
                self.assertEqual((root / "runtime.json").read_bytes(), json.dumps(runtime).encode())
                self.assertEqual(reopened.preview_only, expected)

    def test_nonzero_actual_ack_terminates_failed_startup(self):
        process = Mock()
        process.poll.return_value = None
        bad = dict(frame=1, frame_received_at=100.0, motor=dict(armed=True, ack=[1, 1, 10, 10, 0]))
        with (
            patch.object(installer.subprocess, "Popen", return_value=process),
            patch.object(
                installer, "read_json", side_effect=[bad, dict(active=False, left=0, right=0)]
            ),
            patch.object(installer.Path, "open", unittest.mock.mock_open(read_data=b"")),
            patch.object(installer.Path, "read_text", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "UNO acknowledgement"):
                installer.launch_and_verify(Path("/offline/app"))
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
