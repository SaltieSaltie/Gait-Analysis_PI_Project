import cv2
import time
import os
import argparse
import datetime as dt

import config
from camera_stream import ThreadedCamera
from pose_detector import PoseDetector
from video_recorder import AsyncVideoRecorder
from gait_analyzer import GaitAnalyzer


def draw_status_overlay(frame, status):
    """Afișează statusul aplicației în colțul din stânga sus."""
    color = config.STATUS_COLORS.get(status, (255, 255, 255))
    text = f"STATUS: {status}"

    cv2.rectangle(frame, (10, 10), (340, 60), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (340, 60), color, 2)
    cv2.putText(frame, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def run_video_file_analysis(video_path, detector, analyzer):
    """Procesează un fișier video existent (fără a duplica fișierul video)."""
    if not os.path.exists(video_path):
        print(f"[EROARE] Fișierul video specificat nu există: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(config.DATA_DIR, f"gait_scores_{timestamp}.csv")

    analyzer.reset()
    print(f"[INFO] Se analizează videoclipul: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        person_detected, keypoints, display_frame = detector.process_frame(frame)

        if person_detected:
            analyzer.add_frame_keypoints(keypoints)
            draw_status_overlay(display_frame, "CAPTURING")
        else:
            draw_status_overlay(display_frame, "READY")

        cv2.imshow("Gait Analysis - Video File Mode", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salvează raportul CSV detaliat
    analyzer.export_csv(output_csv_path)


def run_live_camera_analysis(detector, analyzer):
    """Modul Live pe camera web cu înregistrare automată pe detecție subiect."""
    cam = ThreadedCamera(src=config.CAMERA_INDEX, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    recorder = None
    log_saved_timer = None
    current_status = "READY"

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            person_detected, keypoints, display_frame = detector.process_frame(frame)

            if current_status == "READY":
                if person_detected:
                    current_status = "CAPTURING"
                    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

                    video_path = os.path.join(config.DATA_DIR, f"gait_rec_{timestamp}.mp4")
                    h, w, _ = frame.shape
                    recorder = AsyncVideoRecorder(video_path, fps=config.TARGET_FPS, frame_size=(w, h))

                    analyzer.reset()
                    analyzer.add_frame_keypoints(keypoints)

            elif current_status == "CAPTURING":
                if person_detected:
                    recorder.write(frame)
                    analyzer.add_frame_keypoints(keypoints)
                else:
                    recorder.stop()
                    recorder = None

                    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = os.path.join(config.DATA_DIR, f"gait_scores_{timestamp}.csv")
                    analyzer.export_csv(csv_path)

                    current_status = "LOG SAVED"
                    log_saved_timer = time.time()

            elif current_status == "LOG SAVED":
                if time.time() - log_saved_timer >= config.LOG_SAVED_DURATION_SEC:
                    current_status = "READY"

            draw_status_overlay(display_frame, current_status)
            cv2.imshow("Gait Analysis System", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if recorder is not None:
            recorder.stop()
        cam.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gait Analysis Framework (Live & Video Offline)")
    parser.add_argument("-v", "--video", type=str, default=None,
                        help="Calea către un fișier video pentru analiză offline")
    args = parser.parse_args()

    detector = PoseDetector(config.YOLO_MODEL_PATH)
    analyzer = GaitAnalyzer()

    if args.video:
        run_video_file_analysis(args.video, detector, analyzer)
    else:
        run_live_camera_analysis(detector, analyzer)


if __name__ == '__main__':
    main()