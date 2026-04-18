import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mss
import numpy as np

from perception.telemetry import SpeedTelemetryReader, DistTurnKalmanFilter, PixelColorSampler
from .detection import (
    run_detection_once,
    run_lane_segmentation_once,
    apply_temporal_consensus,
    compute_intersection_status,
    resize_to_fit,
)
from .drawing import (
    build_lane_draw_cache,
    draw_lane_from_cache,
    build_sign_overlay_cache,
    build_tl_overlay_cache,
    apply_overlay_cache,
    _draw_legend,
)


def run_inference_screen_capture(args, detector, tl_detector=None, lane_model=None):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")

    has_tl = tl_detector is not None
    has_lane = lane_model is not None
    lane_priority_enabled = has_lane
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold
    lane_conf = args.lane_conf_threshold if args.lane_conf_threshold is not None else args.conf_threshold
    lane_imgsz = args.lane_imgsz if args.lane_imgsz is not None else args.imgsz
    lane_scale = args.lane_inference_scale if args.lane_inference_scale is not None else args.inference_scale
    sign_submit_interval_ms = max(0.0, float(args.sign_submit_interval_ms))
    tl_submit_interval_ms = max(0.0, float(args.tl_submit_interval_ms))

    if lane_priority_enabled:
        if sign_submit_interval_ms <= 0.0:
            sign_submit_interval_ms = 80.0
        if has_tl and tl_submit_interval_ms <= 0.0:
            tl_submit_interval_ms = 80.0

    capture_mode = "async" if args.async_capture else "sync"
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, "
        + (f"tl_conf={tl_conf:.2f}, " if has_tl else "")
        + (f"lane_conf={lane_conf:.2f}, " if has_lane else "")
        + f"imgsz={args.imgsz}, infer_scale={args.inference_scale:.2f}, "
        + (f"lane_imgsz={lane_imgsz}, lane_scale={lane_scale:.2f}, " if has_lane else "")
        + f"mode=async, capture={capture_mode}, lane_priority={lane_priority_enabled}, "
        f"confirm={args.min_confirm_frames}, max_miss={args.max_missing_frames}, "
        f"track_iou={args.track_iou_threshold:.2f}, window={args.window_width}x{args.window_height}"
    )
    if sign_submit_interval_ms > 0.0 or (has_tl and tl_submit_interval_ms > 0.0):
        print(
            f"[INFO] Async submit intervals: sign={sign_submit_interval_ms:.1f}ms"
            + (f", tl={tl_submit_interval_ms:.1f}ms" if has_tl else "")
        )
    if has_tl:
        print("[INFO] Dual-model mode: sign model + traffic-light model running in parallel.")
    if has_lane:
        print("[INFO] Lane segmentation enabled.")

    sct = mss.mss()
    monitor_index = args.monitor_index
    if monitor_index < 1 or monitor_index >= len(sct.monitors):
        fallback_index = 1 if len(sct.monitors) > 1 else 0
        print(f"[WARNING] Invalid monitor index {monitor_index}. Falling back to {fallback_index}.")
        monitor_index = fallback_index

    monitor = sct.monitors[monitor_index]
    print(
        f"Capture monitor {monitor_index}: "
        f"{monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})"
    )

    window_name = f"{args.model} - Real-time Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    # Per-model state
    sign_detections_raw = []
    sign_stable = []
    sign_to_draw = []
    sign_overlay_cache = None
    sign_overlay_shape = None
    sign_tracks = {}
    sign_next_id = 1
    sign_raw_count = 0
    sign_infer_duration = 0.0

    tl_detections_raw = []
    tl_stable = []
    tl_to_draw = []
    tl_overlay_cache = None
    tl_overlay_shape = None
    tl_tracks = {}
    tl_next_id = 1
    tl_raw_count = 0
    tl_infer_duration = 0.0

    lane_instances = []
    lane_draw_cache = None
    lane_draw_cache_shape = None
    lane_infer_duration = 0.0
    lane_intersection_score = 0.0
    lane_intersection = False
    lane_intersection_reasons = []

    # Speed telemetry
    speed_telemetry_reader = None
    speed_kmh = None
    dist_turn_m = None
    dist_dest_m = None
    dist_turn_kalman = DistTurnKalmanFilter(process_noise=2.0, measurement_noise=15.0)
    dist_turn_last_raw: float | None = None
    dist_turn_frozen_since: float | None = None
    dist_turn_frozen_threshold_m: float = 1.0   # metres — smaller change = considered frozen
    dist_turn_frozen_timeout_s: float = 0.35    # seconds before we stop trusting a stuck value
    speed_last_update_ts = 0.0
    speed_stale_timeout_s = 0.50
    if args.speed_telemetry:
        try:
            speed_telemetry_reader = SpeedTelemetryReader(
                process_name="starter.exe",
                module_name="pdd.dll",
                speed_offset=0xE322B0,
                dist_turn_offset=0xF10C70,
                poll_interval_ms=50.0,
            )
            print("[INFO] Speed telemetry enabled.")
        except Exception as exc:
            print(f"[WARNING] Failed to initialize speed telemetry: {exc}")
            speed_telemetry_reader = None

    pixel_sampler = PixelColorSampler(
        pixels=((2283, 93), (2321, 92)),
        target_hex="#D2B819",
        threshold=25.0,
        poll_interval_ms=50.0,
    )

    display_fps_ema = 0.0
    sign_fps_ema = 0.0
    tl_fps_ema = 0.0
    lane_fps_ema = 0.0

    sign_pending = None
    tl_pending = None
    lane_pending = None
    sign_last_submit_time = 0.0
    tl_last_submit_time = 0.0

    sign_worker = ThreadPoolExecutor(max_workers=1)
    tl_worker = ThreadPoolExecutor(max_workers=1) if has_tl else None
    lane_worker = ThreadPoolExecutor(max_workers=1) if has_lane else None

    # Async capture setup
    capture_lock = None
    latest_frame = None
    latest_display_frame = None
    last_async_frame = None
    last_async_display_frame = None
    capture_stop_event = None
    capture_thread = None
    capture_error = None

    if args.async_capture:
        capture_lock = threading.Lock()
        capture_stop_event = threading.Event()

        def capture_loop():
            nonlocal latest_frame, latest_display_frame, capture_error
            local_capture_lock = capture_lock
            assert local_capture_lock is not None
            try:
                with mss.mss() as capture_sct:
                    while not capture_stop_event.is_set():
                        grabbed = np.array(capture_sct.grab(monitor))
                        bgr_frame = cv2.cvtColor(grabbed, cv2.COLOR_BGRA2BGR)
                        disp_base = resize_to_fit(bgr_frame, args.window_width, args.window_height)
                        with local_capture_lock:
                            latest_frame = bgr_frame
                            latest_display_frame = disp_base
            except Exception as exc:
                capture_error = exc
                capture_stop_event.set()

        capture_thread = threading.Thread(target=capture_loop, name="screen-capture-thread", daemon=True)
        capture_thread.start()
        print("[INFO] Async capture enabled (latest-frame mode).")

    try:
        while True:
            loop_start = time.perf_counter()
            stop_requested = False

            # --- Frame acquisition ---
            if not args.async_capture:
                screen_img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
                disp_base = resize_to_fit(frame, args.window_width, args.window_height)
            else:
                assert capture_lock is not None
                frame = None
                disp_base = None

                if capture_error is not None:
                    raise RuntimeError(f"Background capture failed: {capture_error}")

                with capture_lock:
                    if latest_frame is not None:
                        frame = latest_frame
                        disp_base = latest_display_frame
                        latest_frame = None
                        latest_display_frame = None

                if frame is None and last_async_frame is not None:
                    frame = last_async_frame
                    disp_base = last_async_display_frame

                while frame is None:
                    if capture_error is not None:
                        raise RuntimeError(f"Background capture failed: {capture_error}")
                    with capture_lock:
                        if latest_frame is not None:
                            frame = latest_frame
                            disp_base = latest_display_frame
                            latest_frame = None
                            latest_display_frame = None
                    if frame is None:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_requested = True
                            break
                        time.sleep(0.001)

                if stop_requested:
                    break

                last_async_frame = frame
                last_async_display_frame = disp_base

            assert frame is not None
            if disp_base is None:
                disp_base = resize_to_fit(frame, args.window_width, args.window_height)

            disp_frame = disp_base.copy()
            async_input_frame = frame.copy()
            lane_updated = False
            sign_updated = False
            tl_updated = False

            # --- Sign model ---
            if sign_pending is not None and sign_pending.done():
                try:
                    sign_detections_raw, sign_infer_duration = sign_pending.result()
                    sign_raw_count = len(sign_detections_raw)
                    sign_stable, sign_next_id = apply_temporal_consensus(
                        sign_detections_raw, sign_tracks, sign_next_id,
                        args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                    )
                    sign_updated = True
                except Exception as exc:
                    print(f"[ERROR] Sign model inference failed: {exc}")
                    sign_detections_raw = []
                    sign_stable = []
                    sign_tracks.clear()
                    sign_updated = True
                sign_pending = None

            if sign_pending is None:
                now_ts = time.perf_counter()
                if ((now_ts - sign_last_submit_time) * 1000.0) >= sign_submit_interval_ms:
                    sign_pending = sign_worker.submit(
                        run_detection_once, detector, async_input_frame,
                        args.conf_threshold, args.imgsz, args.inference_scale,
                    )
                    sign_last_submit_time = now_ts

            # --- Traffic light model ---
            if has_tl:
                if tl_pending is not None and tl_pending.done():
                    try:
                        tl_detections_raw, tl_infer_duration = tl_pending.result()
                        tl_raw_count = len(tl_detections_raw)
                        tl_stable, tl_next_id = apply_temporal_consensus(
                            tl_detections_raw, tl_tracks, tl_next_id,
                            args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                        )
                        tl_updated = True
                    except Exception as exc:
                        print(f"[ERROR] Traffic light model inference failed: {exc}")
                        tl_detections_raw = []
                        tl_stable = []
                        tl_tracks.clear()
                        tl_updated = True
                    tl_pending = None

                if tl_pending is None:
                    now_ts = time.perf_counter()
                    if ((now_ts - tl_last_submit_time) * 1000.0) >= tl_submit_interval_ms:
                        tl_pending = tl_worker.submit(
                            run_detection_once, tl_detector, async_input_frame,
                            tl_conf, args.imgsz, args.inference_scale,
                        )
                        tl_last_submit_time = now_ts

            # --- Lane model ---
            if has_lane:
                if lane_pending is not None and lane_pending.done():
                    try:
                        lane_instances, lane_infer_duration = lane_pending.result()
                        lane_updated = True
                    except Exception as exc:
                        print(f"[ERROR] Lane model inference failed: {exc}")
                        lane_instances = []
                        lane_updated = True
                    lane_pending = None

                if lane_pending is None:
                    lane_pending = lane_worker.submit(
                        run_lane_segmentation_once, lane_model, async_input_frame,
                        lane_conf, lane_imgsz, lane_scale,
                    )

                if lane_draw_cache is None or lane_updated:
                    lane_draw_cache_shape = None

                if lane_updated:
                    lane_intersection_score, lane_intersection, lane_intersection_reasons = (
                        compute_intersection_status(lane_instances, frame.shape, threshold=args.intersection_threshold)
                    )

            disp_h, disp_w = disp_frame.shape[:2]

            # --- Speed telemetry ---
            if speed_telemetry_reader is not None:
                now_ts = time.perf_counter()
                latest_speed, latest_dist_turn, latest_dist_dest = speed_telemetry_reader.read_telemetry_if_due(now_ts)
                _speed_ms = (speed_kmh / 3.6) if speed_kmh is not None else 0.0
                if latest_speed is not None or latest_dist_turn is not None or latest_dist_dest is not None:
                    if latest_speed is not None:
                        speed_kmh = float(latest_speed)
                        _speed_ms = speed_kmh / 3.6
                    if latest_dist_turn is not None:
                        dist_turn_m = float(latest_dist_turn)
                        # Detect frozen measurement (game stops updating near intersection)
                        if dist_turn_last_raw is None or abs(dist_turn_m - dist_turn_last_raw) >= dist_turn_frozen_threshold_m:
                            dist_turn_last_raw = dist_turn_m
                            dist_turn_frozen_since = now_ts
                        frozen_dur = now_ts - dist_turn_frozen_since if dist_turn_frozen_since is not None else 0.0
                        if frozen_dur < dist_turn_frozen_timeout_s:
                            dist_turn_kalman.update(dist_turn_m, _speed_ms, now_ts)
                        else:
                            # Measurement is stuck — trust only the speed-based prediction
                            dist_turn_kalman.predict(_speed_ms, now_ts)
                    else:
                        dist_turn_kalman.predict(_speed_ms, now_ts)
                    if latest_dist_dest is not None:
                        dist_dest_m = float(latest_dist_dest)
                    speed_last_update_ts = now_ts
                elif speed_kmh is not None and (now_ts - speed_last_update_ts) > speed_stale_timeout_s:
                    speed_kmh = None
                    dist_turn_m = None
                    dist_dest_m = None
                    dist_turn_last_raw = None
                    dist_turn_frozen_since = None
                    dist_turn_kalman.reset()
                else:
                    dist_turn_kalman.predict(_speed_ms, now_ts)

            # --- Draw lane overlay ---
            if has_lane:
                if lane_draw_cache is None or lane_updated or lane_draw_cache_shape != (disp_h, disp_w):
                    lane_draw_cache = build_lane_draw_cache(lane_instances, disp_frame.shape)
                    lane_draw_cache_shape = (disp_h, disp_w)
                draw_lane_from_cache(disp_frame, lane_draw_cache, alpha=args.lane_mask_alpha)

            # --- Draw sign detections ---
            sign_to_draw = sign_stable[:args.max_draw_detections] if args.max_draw_detections > 0 else sign_stable
            if sign_overlay_cache is None or sign_updated or sign_overlay_shape != (disp_h, disp_w):
                sign_overlay_cache = build_sign_overlay_cache(disp_frame.shape, frame.shape, sign_to_draw, detector)
                sign_overlay_shape = (disp_h, disp_w)
            apply_overlay_cache(disp_frame, sign_overlay_cache)

            # --- Draw traffic light detections ---
            if has_tl:
                tl_to_draw = tl_stable[:args.max_draw_detections] if args.max_draw_detections > 0 else tl_stable
                if tl_overlay_cache is None or tl_updated or tl_overlay_shape != (disp_h, disp_w):
                    tl_overlay_cache = build_tl_overlay_cache(disp_frame.shape, frame.shape, tl_to_draw, tl_detector)
                    tl_overlay_shape = (disp_h, disp_w)
                apply_overlay_cache(disp_frame, tl_overlay_cache)

            # --- HUD ---
            loop_duration = max(time.perf_counter() - loop_start, 1e-6)
            display_fps = 1.0 / loop_duration
            sign_fps = 1.0 / sign_infer_duration if sign_infer_duration > 0.0 else 0.0
            tl_fps = 1.0 / tl_infer_duration if tl_infer_duration > 0.0 else 0.0
            lane_fps = 1.0 / lane_infer_duration if lane_infer_duration > 0.0 else 0.0

            display_fps_ema = display_fps if display_fps_ema == 0.0 else 0.9 * display_fps_ema + 0.1 * display_fps
            if sign_fps > 0.0:
                sign_fps_ema = sign_fps if sign_fps_ema == 0.0 else 0.9 * sign_fps_ema + 0.1 * sign_fps
            if tl_fps > 0.0:
                tl_fps_ema = tl_fps if tl_fps_ema == 0.0 else 0.9 * tl_fps_ema + 0.1 * tl_fps
            if lane_fps > 0.0:
                lane_fps_ema = lane_fps if lane_fps_ema == 0.0 else 0.9 * lane_fps_ema + 0.1 * lane_fps

            cv2.putText(
                disp_frame,
                f"Display FPS: {display_fps_ema:.1f} | Sign FPS: {sign_fps_ema:.1f}"
                + (f" | TL FPS: {tl_fps_ema:.1f}" if has_tl else "")
                + (f" | Lane FPS: {lane_fps_ema:.1f}" if has_lane else ""),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2,
            )
            cv2.putText(
                disp_frame,
                f"Mode: ASYNC | Scale: {args.inference_scale:.2f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            cv2.putText(
                disp_frame,
                f"Signs raw/stable: {sign_raw_count}/{len(sign_to_draw)}"
                + (f"  |  TL raw/stable: {tl_raw_count}/{len(tl_to_draw)}" if has_tl else ""),
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
            )
            if has_lane:
                status_text = f"Intersection: {'YES' if lane_intersection else 'no'} ({lane_intersection_score:.2f})"
                if lane_intersection_reasons:
                    status_text += " | " + ",".join(lane_intersection_reasons)
                cv2.putText(
                    disp_frame, status_text,
                    (20, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (20, 20, 220) if lane_intersection else (80, 180, 80), 2,
                )
            if speed_telemetry_reader is not None:
                speed_text = f"Speed(mem): {speed_kmh:.1f} km/h" if speed_kmh is not None else "Speed(mem): --"
                _kf_est = dist_turn_kalman.estimate
                dist_text = f"Dist turn: {dist_turn_m:.0f} m" if dist_turn_m is not None else "Dist turn: --"
                kf_text = f"(KF: {_kf_est:.0f} m)" if _kf_est is not None else "(KF: --)"
                dest_text = f"Dist dest: {dist_dest_m:.0f} m" if dist_dest_m is not None else "Dist dest: --"
                cv2.putText(disp_frame, speed_text + "  |  " + dist_text + " " + kf_text + "  |  " + dest_text, (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            pixel_sampler.sample_if_due(time.perf_counter())
            if pixel_sampler.last_results:
                left_match  = pixel_sampler.last_results[0][2]
                right_match = pixel_sampler.last_results[1][2]
                left_hex  = "#{:02X}{:02X}{:02X}".format(*pixel_sampler.last_results[0][4])
                right_hex = "#{:02X}{:02X}{:02X}".format(*pixel_sampler.last_results[1][4])
                if left_match and right_match:
                    nav_label = "turn around"
                elif right_match:
                    nav_label = "turn left"
                elif left_match:
                    nav_label = "turn right"
                else:
                    nav_label = "move forward"
                px_color = (0, 220, 80) if (left_match or right_match) else (160, 160, 160)
                cv2.putText(
                    disp_frame,
                    f"Nav: {nav_label}  [{left_hex} | {right_hex}]",
                    (20, 207), cv2.FONT_HERSHEY_SIMPLEX, 0.60, px_color, 2,
                )

            _draw_legend(disp_frame, has_tl, has_lane)

            cv2.imshow(window_name, disp_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if capture_stop_event is not None:
            capture_stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
        sign_worker.shutdown(wait=False, cancel_futures=True)
        if tl_worker is not None:
            tl_worker.shutdown(wait=False, cancel_futures=True)
        if lane_worker is not None:
            lane_worker.shutdown(wait=False, cancel_futures=True)
        if speed_telemetry_reader is not None:
            speed_telemetry_reader.close()
        cv2.destroyAllWindows()
