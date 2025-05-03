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
    input_video_path = r"C:\Users\user\Desktop\projects\Volleyball_detector\videos_volley\17 - Trim.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='weights/best_for_players_model.pt')
    ball_tracker = BallTracker(model_path='weights/best_for_ball_models.pt')
    # player_detections = player_tracker.detect_frames(video_frames,
    #                                                  read_from_stub=True,
    #                                                  stub_path=r"tracker_stubs/player_detections.pkl"
    #                                                  )
    # ball_detections = ball_tracker.detect_frames(video_frames,
    #                                                  read_from_stub=False,
    #                                                  stub_path=r"tracker_stubs/ball_detections.pkl"
    #                                                  )
    player_detections = player_tracker.track(video_frames, show=False, tracker=r"C:\Users\user\Desktop\projects\Volleyball_detector\botsort_players.yaml")
    ball_detections = ball_tracker.track(video_frames, show=False, tracker=r"bytetrack.yaml")
    
    #Court Line Detector model
    court_model_path = r"weights/best_for_court_models.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[74])

    #MiniCourt
    mini_court = MiniCourt(video_frames[5])
    
    frames_with_players = player_tracker.draw_bboxes(video_frames, player_detections)
    
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

    
    save_video(output_video_frames, r"output_videos/output_video9.avi")
    
if __name__ == "__main__":
    main()