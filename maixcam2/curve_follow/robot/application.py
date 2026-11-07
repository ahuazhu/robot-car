"""Parametric curve following. Preview is the default; current-pose calibration required."""

import os, time, json
from contextlib import ExitStack
import cv2
import numpy as np
from robot.navigation.tracker import TargetTracker
from robot.interface.road_overlay import render_target
from robot.control.drive_link import DriveLink, PreviewLink
from robot.control.motor_channel import MotorChannel
from robot.control.controller import CommandRamp, compensate_pwm, RoadSupervisor, motion_from_pwm
from robot.control.session import DrivingSession, FrameDecision
from robot.config import RuntimeSettings
from robot.interface.preview_transport import LiveStream
from robot.navigation.steering_control import SteeringControl
from robot.navigation.road_recovery import RoadSearch
from robot.tasks.start_signal import StartSignal
from robot.tasks.lap_counter import LapMarkerDetector, LapCounter
from robot.interface.screen_view import DeviceScreen
from robot.core.native_geometry import native_fit_status
from robot.tasks.crosswalk import CrosswalkDetector, CrosswalkTask, draw_crosswalk_overlay


def main(app_dir):
    from maix import app, camera, display, image, touchscreen

    settings = RuntimeSettings.load(app_dir)
    cal, runtime = settings.calibration, settings.runtime
    policy, crosswalk_policy = settings.road_policy, settings.crosswalk_policy
    road_cal = settings.road_calibration
    control_frame = runtime.get("control_frame", {})
    crosswalk_task = CrosswalkTask(crosswalk_policy)
    start_light = StartSignal(runtime.get("start_signal"), app_dir=app_dir)
    lap_config = runtime.get("race", {})
    laps = LapCounter(
        lap_config,
        float(crosswalk_policy.body_front_m or 0.02),
        float(crosswalk_policy.body_rear_m or 0.16),
        float(crosswalk_policy.body_half_width_m or 0.085),
        reference_origin_m=control_frame.get("origin_m", [0.0, 0.0]),
    )
    if os.environ.get("START_LIGHT_DISARMED") == "1":
        start_light.cancel()
    calibrated, pwm_boost = settings.calibrated, settings.pwm_boost
    vision_budget_ms, turn_lever = settings.vision_budget_ms, settings.turn_lever
    steering = SteeringControl(
        policy,
        control_frame.get("origin_m", [0.0, 0.0]),
        control_frame.get("body_m"),
        float(cal["road_width_m"]),
        turn_lever,
        settings.steering_gain,
        settings.cruise_pwm,
        float(crosswalk_policy.brake_deceleration_m_s2 or 0.08),
        crosswalk_policy.speed_per_pwm_m_s,
        calibration=cal,
        response_s=crosswalk_policy.speed_response_s,
        straight_cruise=settings.straight_cruise_pwm,
        confirmed_cruise=settings.confirmed_cruise_pwm,
    )
    search = RoadSearch(
        policy,
        steering.origin,
        steering.body,
        steering.width,
        settings.search_cruise_pwm,
        crosswalk_policy.speed_response_s,
        allow_visual_scan=runtime.get("road_search", {}).get("allow_visual_scan", False),
    )
    backend = native_fit_status()
    print("GEOMETRY_BACKEND", json.dumps(backend), flush=True)
    cv2.setNumThreads(1)
    duration, preview_only = settings.duration_s, settings.preview_only
    link = PreviewLink() if preview_only else MotorChannel(DriveLink())
    stream = cam = None
    was_pressed = False
    next_log = next_stream = next_display = 0.0
    last_display_state = None
    last_tracking_state = previous_captured = None
    began = time.monotonic()
    frames = observed_frames = 0
    saved, stats = set(), {}
    ramp, recovery = CommandRamp(), RoadSupervisor(policy.missing_command_s)
    try:
        stream = LiveStream(8000)
        session = DrivingSession(
            settings,
            link,
            stream.control,
            steering,
            search,
            start_light,
            laps,
            crosswalk_task,
            ramp,
            recovery,
        )
        start_light.start()
        session.connect()
        vision = TargetTracker(road_cal)
        crosswalk_detector = CrosswalkDetector(vision.pixels, crosswalk_policy)
        lap_detector = LapMarkerDetector(vision.pixels, lap_config)
        cam = camera.Camera(320, 240, image.Format.FMT_BGR888)
        screen = display.Display()
        touch = touchscreen.TouchScreen()
        touch.clear()
        device_screen = DeviceScreen(screen.width(), screen.height())
        print("DEVICE_SCREEN", json.dumps(device_screen.snapshot()), flush=True)
        print(
            "CURVE FOLLOW v0.6.75 | ESP32 start signal | preview_only=%s | calibrated=%s | no session time limit"
            % (preview_only, calibrated),
            flush=True,
        )
        began = time.monotonic()  # Exclude camera startup from test/debug elapsed time.
        while not app.need_exit():
            now = time.monotonic()
            if duration and now - began >= duration:
                break
            read_began = time.monotonic()
            frame = cam.read()
            camera_read_at = time.monotonic()
            bgr = image.image2cv(frame, ensure_bgr=True, copy=True)
            captured = time.monotonic()
            profile_ms = dict(
                camera_read=round((camera_read_at - read_began) * 1000, 2),
                image_conversion=round((captured - camera_read_at) * 1000, 2),
                frame_period=None
                if previous_captured is None
                else round((captured - previous_captured) * 1000, 2),
            )
            previous_captured = captured
            active, pwm = session.active, session.pwm
            v, omega = motion_from_pwm(pwm, turn_lever) if active else (0.0, 0.0)
            # Arming uses a trusted current component; driving keeps bounded
            # historical support. Both queries share the same persistent model.
            result, details = vision.update(bgr, captured, v, omega, require_current=not active)
            frames += 1
            motor_before = None if preview_only else link.snapshot()
            lap_tick = time.perf_counter()
            lap_observation = lap_detector.detect(
                bgr,
                details,
                details.get("measured_road") or result,
                captured,
                motion=details.get("ground_motion"),
            )
            lap_finished = laps.update(
                lap_observation,
                details.get("ground_motion"),
                captured,
                active,
            )
            if laps.event:
                print("LAP EVENT", json.dumps(laps.snapshot()), flush=True)
            profile_ms["lap_counter"] = round((time.perf_counter() - lap_tick) * 1000, 2)
            crosswalk_tick = time.perf_counter()
            crossing = crosswalk_detector.detect(
                details, details.get("measured_road") or result, captured
            )
            actual_ack = motor_before.get("ack", []) if motor_before else []
            zero_ack = bool(
                motor_before
                and motor_before.get("ack_age_ms", 1000) < 200
                and len(actual_ack) >= 4
                and actual_ack[1:4] == [1, 0, 0]
            )
            lamp_on = bool(
                motor_before
                and motor_before.get("brake_lamp_on")
                and motor_before.get("brake_lamp_supported")
            )
            crosswalk_state = crosswalk_task.update(
                crossing,
                details.get("ground_motion"),
                captured,
                active,
                zero_ack,
                lamp_on,
                bool(
                    result.get("valid")
                    and result.get("observed")
                    and result.get("state") == "OBSERVED"
                ),
            )
            crosswalk_ms = (time.perf_counter() - crosswalk_tick) * 1000
            stats[result["mode"]] = stats.get(result["mode"], 0) + 1
            observed_frames += int(result["observed"])
            steering_tick = time.monotonic()
            steering.observe(details["ground_motion"], captured, allow_feedback=active)
            command = steering.command(result, age_s=time.monotonic() - captured)
            if (
                command is None
                and (steering.status.get("envelope") or {}).get("first_unsafe_m") == 0
            ):
                result["invalid_reason"] = "BODY ENVELOPE BLOCKED"
            if command is not None:
                # Tracker's nearest point remains an association/validity anchor;
                # public target fields describe the actual selected pursuit goal.
                result = dict(
                    result,
                    nearest_reference_m=result["control_target_m"],
                    control_target_m=steering.status["receive_target_m"],
                    target_m=(
                        np.asarray(steering.status["receive_target_m"]) + steering.origin
                    ).tolist(),
                    target_selection=steering.status["target_selection"],
                    target_distance_m=steering.status["target_distance_m"],
                    target_speed_scale=steering.status["target_speed_scale"],
                    target_inferred=not steering.status["target_observed"],
                    control_target_source="OBSERVED"
                    if steering.status["target_observed"]
                    else "HISTORY PREDICTION",
                )
            steering_done = time.monotonic()
            search_tick = steering_done
            # START resets this map; inactive observations cannot be reused.
            if active:
                search.observe(
                    result, details["ground_motion"], captured, surface=details.get("surface")
                )
            else:
                search.observe_hint(result)
            follow_command = command
            search_allowed = crosswalk_task.track is None or (
                crosswalk_task.phase == "PASSING" and crosswalk_task.pending_track is None
            )
            command = search.command(
                result,
                command,
                details["ground_motion"],
                captured,
                active=active,
                allowed=search_allowed,
                stopped=zero_ack,
            )
            profile_ms["road_search"] = round((time.monotonic() - search_tick) * 1000, 2)
            profile_ms.update(result["timings_ms"])
            profile_ms["crosswalk"] = round(crosswalk_ms, 2)
            profile_ms["steering"] = round((steering_done - steering_tick) * 1000, 2)
            detection_ms = (time.monotonic() - captured) * 1000
            lamp_start = start_light.update(captured, active=active, timely=detection_ms <= vision_budget_ms)
            profile_ms["control_pipeline"] = round(detection_ms, 2)
            supervisor_tick = time.monotonic()
            x, y, pressed = touch.read()
            action = stream.control.take()
            touch_action = device_screen.action_at(x, y) if pressed and not was_pressed else None
            was_pressed = pressed
            if touch_action == "exit":
                break
            decision = FrameDecision(
                result=result,
                details=details,
                captured=captured,
                command=command,
                follow_command=follow_command,
                processing_ms=detection_ms,
                lap_observation=lap_observation,
                lap_finished=lap_finished,
                motor_snapshot=motor_before,
                zero_ack=zero_ack,
                crosswalk_state=crosswalk_state,
                light_start=lamp_start,
            )
            session.update(decision, (action, touch_action), bgr)
            command = decision.command
            active, pwm, reason, link_ok = (
                session.active,
                session.pwm,
                session.reason,
                session.link_ok,
            )
            readiness = dict(
                session.readiness(decision),
                frame=frames,
                observed_streak=session.consecutive,
                required_observations=policy.startup_observations,
                current_command_timely=detection_ms <= vision_budget_ms,
                processing_ms=round(detection_ms, 1),
                vision_budget_ms=vision_budget_ms,
                confidence=result["confidence"],
                source=result["mode"],
            )
            ready = readiness["direct_start_ready"] and not (
                start_light.enabled and start_light.armed
            )
            applied_pwm = compensate_pwm(pwm, pwm_boost)
            total_pwm = abs(applied_pwm[0]) + abs(applied_pwm[1])
            steering.status["published_curvature_proxy_m_inv"] = (
                (applied_pwm[1] - applied_pwm[0]) / (turn_lever * total_pwm) if total_pwm else 0.0
            )
            readiness["recovery"] = dict(
                recovery.snapshot(), search=search.snapshot(), preview_error=stream.preview_error
            )
            readiness["laps"] = dict(laps.snapshot(), detector=lap_detector.status)
            readiness["minimum_road_quality"] = policy.measurement_quality
            readiness["start_light"] = start_light.snapshot()
            readiness["steering"] = steering.status
            readiness["crosswalk"] = dict(
                crosswalk_task.snapshot(captured), detection=crosswalk_detector.status
            )
            stream.control.update(active, ready, reason, applied_pwm, readiness=readiness)
            stream.metrics = dict(
                device_screen=device_screen.snapshot(),
                frame=frames,
                app_version="0.6.75",
                geometry_backend=backend,
                frame_received_at=captured,
                read_ms=round((captured - read_began) * 1000, 1),
                calibrated=calibrated,
                preview_only=preview_only,
                laps=dict(laps.snapshot(), detector=lap_detector.status),
                start_light=start_light.snapshot(),
                motor=None if preview_only else link.snapshot(),
                detection_ms=round(detection_ms, 1),
                recovery=dict(recovery.snapshot(), search=search.snapshot()),
                preview_error=stream.preview_error,
                readiness=readiness,
                crosswalk=dict(
                    crosswalk_task.snapshot(captured), detection=crosswalk_detector.status
                ),
                crosswalk_ms=round(crosswalk_ms, 2),
                target=dict(
                    {k: v for k, v in result.items() if k != "path"},
                    road_path_m=[[round(x, 4), round(y, 4)] for x, y in result.get("path", [])],
                ),
                steering=steering.status,
            )
            profile_ms["supervision_motor_telemetry"] = round(
                (time.monotonic() - supervisor_tick) * 1000, 2
            )
            preview_now = time.monotonic()
            display_state = (active, start_light.armed, bool(any(applied_pwm)))
            display_due = preview_now >= next_display or display_state != last_display_state
            stream_due = preview_now >= next_stream
            debug_seconds = (
                [second for second in (2, 4, 6) if preview_now - began >= second and second not in saved]
                if os.environ.get("MAIXPY_DEBUG_IMAGES") == "1" else []
            )
            profile_ms.update(
                overlay=0.0, display_compose=0.0, display_conversion=0.0,
                display_show=0.0, stream_publish=0.0,
            )
            # Control and touch handling run on every frame. Presentation has
            # its own cadence and never catches up by drawing old frames.
            if display_due or stream_due or debug_seconds:
                overlay_tick = time.monotonic()
                # The red marker must show the actual pursuit goal in the received
                # image's coordinate frame, not the nearest geometric anchor.
                display_result = dict(result, target_m=None)
                if command is not None and steering.status.get("receive_target_m") is not None:
                    display_result["target_m"] = (
                        np.asarray(steering.status["receive_target_m"]) + steering.origin
                    ).tolist()
                out, bird = render_target(bgr, display_result, details, cal)
                draw_crosswalk_overlay(out, crosswalk_task.snapshot(captured), cal)
                lap_detector.draw(out)
                if laps.enabled:
                    cv2.putText(
                        out, "LAPS %d/%d" % (laps.count, laps.total), (4, 69), 0, 0.45, (0, 255, 255), 1
                    )
                raw_preview = bgr.copy()
                if not calibrated:
                    cv2.putText(out, "REFERENCE GEOMETRY ONLY", (4, 200), 0, 0.40, (0, 0, 255), 1)
                profile_ms["overlay"] = round((time.monotonic() - overlay_tick) * 1000, 2)
                if display_due:
                    display_tick = time.monotonic()
                    screen_out = device_screen.compose(
                        out,
                        reason,
                        applied_pwm,
                        frames,
                        arm_ready=readiness["arm_start_light_ready"],
                        armed=start_light.armed,
                    )
                    display_composed = time.monotonic()
                    screen_image = image.cv2image(screen_out, bgr=True, copy=True)
                    display_ready = time.monotonic()
                    screen.show(screen_image)
                    displayed_at = time.monotonic()
                    profile_ms["display_compose"] = round((display_composed - display_tick) * 1000, 2)
                    profile_ms["display_conversion"] = round((display_ready - display_composed) * 1000, 2)
                    profile_ms["display_show"] = round((displayed_at - display_ready) * 1000, 2)
                    # Schedule from completion, including slow SDK calls; never
                    # spend the next control frame repaying a display backlog.
                    next_display = displayed_at + 0.20
                    last_display_state = display_state
                if stream_due:
                    stream_tick = time.monotonic()
                    stream.write(np.hstack((raw_preview, out, bird)))
                    next_stream = time.monotonic() + 0.10
                    profile_ms["stream_publish"] = round((time.monotonic() - stream_tick) * 1000, 2)
            now = time.monotonic()
            log_tick = time.monotonic()
            if (
                now >= next_log
                or result["state"] == "CONFIRMING"
                or result["state"] != last_tracking_state
            ):
                print(
                    "CONTROL",
                    json.dumps(
                        {
                            "active": active,
                            "reason": reason,
                            "recovery": dict(recovery.snapshot(), search=search.snapshot()),
                            "target": {k: v for k, v in result.items() if k != "path"},
                            "detection_ms": round(detection_ms, 1),
                            "loop_processing_ms": round((time.monotonic() - captured) * 1000, 1),
                            "pwm": applied_pwm,
                            "logical_pwm": pwm,
                            "frame": frames,
                            "frame_received_at": captured,
                            "read_ms": round((captured - read_began) * 1000, 1),
                            "uno_connected": None if preview_only else link_ok,
                            "calibrated": calibrated,
                            "preview_only": preview_only,
                            "crosswalk": dict(
                                crosswalk_task.snapshot(captured),
                                detection=crosswalk_detector.status,
                            ),
                            "start_light": start_light.snapshot(),
                            "laps": laps.snapshot(),
                            "steering": steering.status,
                            "profile_ms": profile_ms,
                        }
                    ),
                    flush=True,
                )
                next_log = now + 0.5
            profile_ms["logging"] = round((time.monotonic() - log_tick) * 1000, 2)
            profile_ms["frame_processing"] = round((time.monotonic() - captured) * 1000, 2)
            stream.metrics["profile_ms"] = profile_ms
            readiness["profile_ms"] = profile_ms
            last_tracking_state = result["state"]
            for second in debug_seconds:
                folder = os.path.join(os.environ["MAIXPY_SKILL_RUN_DIR"], "artifacts")
                os.makedirs(folder, exist_ok=True)
                for label, picture in [("raw", bgr), ("control", out)]:
                    path = os.path.join(folder, "%s-%d.jpg" % (label, second))
                    cv2.imwrite(path, picture)
                    print("MAIXPY_SKILL_IMAGE", path, flush=True)
                saved.add(second)
    except Exception as exc:
        print("CONTROL FAULT", repr(exc), flush=True)
        raise
    finally:
        print(
            "TARGET_SUMMARY",
            json.dumps(dict(frames=frames, observed=observed_frames, modes=stats)),
            flush=True,
        )
        with ExitStack() as cleanup:
            cleanup.callback(link.port.close)
            cleanup.callback(start_light.close)
            if cam:
                cleanup.callback(cam.close)
            if stream:
                cleanup.callback(stream.stop)
            print("FINAL_STOP_ACK", link.stop(), flush=True)
        print("TARGET FOLLOW EXIT", flush=True)
