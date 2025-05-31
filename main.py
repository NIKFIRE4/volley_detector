from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
from trackers import PlayerTracker,BallTracker
from court_line_detection import CourtLineDetector
import cv2
import pandas as pd
from copy import deepcopy
from mini_court import MiniCourt


def main():
    # Read Video
    input_video_path = r"videos_volley\17 - Trim.mp4"
    all_ball_detections = []
    cap = cv2.VideoCapture(input_video_path)
    output_frames = []
    
    # Инициализация моделей один раз (вне цикла)
    player_tracker = PlayerTracker(model_path='weights/best_for_players_model.pt')
    ball_tracker = BallTracker(model_path='weights/best_for_ball_models.pt')
    court_model_path = r"weights/best_for_court_models.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    # Получаем первый кадр для инициализации MiniCourt
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка чтения видео")
        return
    
    mini_court = MiniCourt(first_frame)
    
    # Сбрасываем видео в начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Создаем объект для записи видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r"output_videos/output_video11.avi", fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

    # Detect Players and Ball
        player_detections = player_tracker.track([frame],frame_idx=frame_count, show=False,
                                                tracker=r"botsort_players.yaml",
                                                read_from_stub=False,
                                                stub_path=r"tracker_stubs/player_detections.pkl"
                                                )
        ball_detections = ball_tracker.track([frame], frame_idx=frame_count, show=False, 
                                            tracker=r"bytetrack_ball.yaml", 
                                            read_from_stub=False, 
                                            stub_path=r"tracker_stubs/ball_detections.pkl"
                                            )
        all_ball_detections.append(ball_detections[0])
        
        #Court Line Detector model
        court_keypoints = court_line_detector.predict(frame)
        
        frames_with_players = player_tracker.draw_bboxes([frame], player_detections)
        
        frames_with_players_and_ball = ball_tracker.draw_bboxes(
            frames_with_players,
            ball_detections
        )

        # Draw court keypoints
        output_video_frames = court_line_detector.draw_keypoints_on_video(
            frames_with_players_and_ball,
            court_keypoints
        )
        output_video_frames = mini_court.draw_mini_court(output_video_frames)

        # Добавляем номер кадра
        cv2.putText(output_video_frames[0], f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Записываем обработанный кадр в выходное видео
        out.write(output_video_frames[0])
        
        frame_count += 1
        print(f"Обработано кадров: {frame_count}")
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    ball_shot_frames = ball_tracker.get_ball_shot_frames(all_ball_detections)
    print(ball_shot_frames)
    print(f"Обработка завершена. Всего кадров: {frame_count}")
    
if __name__ == "__main__":
    main()