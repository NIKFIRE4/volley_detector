from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def track(self, frames, show=False, tracker=None):
        return self.model.track(frames, show=show, tracker=tracker)

    # def choose_and_filter_players(self, court_keypoints, player_detections):
    #     player_detections_first_frame = player_detections[0]
    #     chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
    #     filtered_player_detections = []
    #     for player_dict in player_detections:
    #         filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
    #         filtered_player_detections.append(filtered_player_dict)
    #     return filtered_player_detections

    # def choose_players(self, court_keypoints, player_dict):
    #     distances = []
    #     for track_id, bbox in player_dict.items():
    #         player_center = get_center_of_bbox(bbox)

    #         min_distance = float('inf')
    #         for i in range(0,len(court_keypoints),2):
    #             court_keypoint = (court_keypoints[i], court_keypoints[i+1])
    #             distance = measure_distance(player_center, court_keypoint)
    #             if distance < min_distance:
    #                 min_distance = distance
    #         distances.append((track_id, min_distance))
        
    #     # sorrt the distances in ascending order
    #     distances.sort(key = lambda x: x[1])
    #     # Choose the first 2 tracks
    #     chosen_players = [distances[0][0], distances[1][0]]
    #     return chosen_players


    # def detect_frames(self,frames, read_from_stub=False, stub_path=None):
    #     player_detections = []

    #     if read_from_stub and stub_path is not None:
    #         with open(stub_path, 'rb') as f:
    #             player_detections = pickle.load(f)
    #         return player_detections

    #     for frame in frames:
    #         player_dict = self.detect_frame(frame)
    #         player_detections.append(player_dict)
        
    #     if stub_path is not None:
    #         with open(stub_path, 'wb') as f:
    #             pickle.dump(player_detections, f)
        
    #     return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "player":
                player_dict[track_id] = result
        
        return player_dict

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

    