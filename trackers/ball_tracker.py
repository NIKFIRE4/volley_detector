from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def track(self, frames, show=False, tracker=None):
        return self.model.track(frames, show=show, tracker=tracker)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    # def get_ball_shot_frames(self,ball_positions):
    #     ball_positions = [x.get(1,[]) for x in ball_positions]
    #     # convert the list into pandas dataframe
    #     df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

    #     df_ball_positions['ball_hit'] = 0

    #     df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
    #     df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    #     df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
    #     minimum_change_frames_for_hit = 25
    #     for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
    #         negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
    #         positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

    #         if negative_position_change or positive_position_change:
    #             change_count = 0 
    #             for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
    #                 negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
    #                 positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

    #                 if negative_position_change and negative_position_change_following_frame:
    #                     change_count+=1
    #                 elif positive_position_change and positive_position_change_following_frame:
    #                     change_count+=1
            
    #             if change_count>minimum_change_frames_for_hit-1:
    #                 df_ball_positions['ball_hit'].iloc[i] = 1

    #     frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

    #     return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict


    def draw_bboxes(self, frames, results_list):
        """
        Рисует bounding boxes на кадрах видео.
        
        Args:
            frames (List[np.ndarray]): Список кадров видео.
            results_list (List[ultralytics.engine.results.Results]): Результаты отслеживания YOLO.
            
        Returns:
            List[np.ndarray]: Кадры с нарисованными bounding boxes.
        """
        output_frames = []

        for frame, results in zip(frames, results_list):
            annotated_frame = frame.copy()

            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты bounding box
                    track_id = int(box.id.item()) if box.id is not None else None  # ID трека
                    conf = float(box.conf)  # Уверенность

                    # Рисуем bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # (Опционально) Пишем ID трека и уверенность
                    label = f"{track_id}: {conf:.2f}" if track_id is not None else f"{conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            output_frames.append(annotated_frame)

        return output_frames
