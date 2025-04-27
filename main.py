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



def main():
    # Read Video
    input_video_path = r"C:\Users\user\Desktop\projects\Volleyball_detector\videos_volley\17 - Trim.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='weights/best_for_players_model.pt')
    ball_tracker = BallTracker(model_path='weights/best_for_ball_models.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path=r"tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path=r"tracker_stubs/ball_detections.pkl"
                                                     )
    
    #Court Line Detector model
    court_model_path = r"weights/best_for_court_models.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[74])
    #ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    #hit_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    #print(f"Ball hit detected on frames: {hit_frames}")

    # Show frames where ball was hit
    # for frame_idx in hit_frames:
    #     cv2.imshow(f"Ball Hit - Frame {frame_idx}", video_frames[frame_idx])
    #     cv2.waitKey(0)  # Wait until user presses a key to move to the next frame

    # cv2.destroyAllWindows()  # Close all OpenCV windows
    #draw player and ball boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    #draw court keypoints
    
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    save_video(output_video_frames, r"output_videos/output_video7.avi")
    
if __name__ == "__main__":
    main()