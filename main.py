from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters)
from trackers import PlayerTracker, BallTracker
from court_line_detection import CourtLineDetector
import cv2
import pandas as pd
from copy import deepcopy
from mini_court import MiniCourt


def main():
    input_video_path = r"videos_volley\Brovkina - Trim111.mp4"
    all_ball_detections = []

    cap = cv2.VideoCapture(input_video_path)

    # ── Инициализация моделей ──────────────────────────────────────────
    player_tracker = PlayerTracker(model_path='weights/best_for_players_model.pt')
    ball_tracker   = BallTracker(model_path='weights/best_for_ball_models.pt')
    court_line_detector = CourtLineDetector(r"weights/best_for_court_models.pth")

    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка чтения видео")
        return
    mini_court = MiniCourt(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Параметры выходного видео ──────────────────────────────────────
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r"output_videos/output_video11.avi", fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Трекинг игроков ───────────────────────────────────────────
        player_results = player_tracker.track(
            [frame], frame_idx=frame_count, show=False,
            tracker=r"botsort_players.yaml",
            read_from_stub=False,
            stub_path=r"tracker_stubs/player_detections.pkl"
        )
        # Строим стабильные ID из результатов YOLO
        # player_results — список из одного Results-объекта
        stable_player_dict = player_tracker.build_stable_detections(
            frame, player_results[0] if isinstance(player_results, list) else player_results
        )

        # ── Трекинг мяча ──────────────────────────────────────────────
        ball_results = ball_tracker.track(
            [frame], frame_idx=frame_count, show=False,
            tracker=r"bytetrack_ball.yaml",
            read_from_stub=False,
            stub_path=r"tracker_stubs/ball_detections.pkl"
        )
        ball_res_single = ball_results[0] if isinstance(ball_results, list) else ball_results
        all_ball_detections.append(ball_res_single)

        # Обновляем буфер траектории
        ball_tracker.update_trail(ball_res_single)

        # ── Детекция линий корта ──────────────────────────────────────
        court_keypoints = court_line_detector.predict(frame)

        # ── Отрисовка ─────────────────────────────────────────────────
        # 1) Игроки со стабильными ID
        frame_with_players = player_tracker.draw_bboxes(
            [frame], [stable_player_dict]
        )[0]

        # 2) Мяч (bbox)
        frame_with_ball = ball_tracker.draw_bboxes(
            [frame_with_players], [ball_res_single]
        )[0]

        # 3) Траектория мяча (поверх bbox)
        frame_with_trail = ball_tracker.draw_trail(frame_with_ball)

        # 4) Ключевые точки корта
        output_frames = court_line_detector.draw_keypoints_on_video(
            [frame_with_trail], court_keypoints
        )

        # 5) Мини-корт
        output_frames = mini_court.draw_mini_court(output_frames)

        # 6) Номер кадра
        cv2.putText(output_frames[0], f"Frame: {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(output_frames[0])
        frame_count += 1
        print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ball_shot_frames = ball_tracker.get_ball_shot_frames(all_ball_detections)
    print("Кадры с ударами:", ball_shot_frames)
    print(f"Обработка завершена. Всего кадров: {frame_count}")


if __name__ == "__main__":
    main()