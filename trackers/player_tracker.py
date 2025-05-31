from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.all_detections = {}  # Словарь для хранения всех детекций

    def track(self, frames, frame_idx=0, show=False, tracker=None,
              read_from_stub=False, stub_path=None):
        # 1) Попытаться загрузить из stub
        if read_from_stub and stub_path and not self.all_detections:
            with open(stub_path, 'rb') as f:
                self.all_detections = pickle.load(f)

        # Если уже есть результаты для данного frame_idx, вернуть их сразу
        if frame_idx in self.all_detections:
            return self.all_detections[frame_idx]

        # 2) Выполнить трекинг
        tracked_list = self.model.track(frames, show=show, tracker=tracker)
        # Здесь предполагаем tracked_list — список результатов для каждого кадра

        # 3) Записать в словарь с правильными ключами
        for offset, detection in enumerate(tracked_list):
            idx = frame_idx + offset
            self.all_detections[idx] = detection

        # Пересохранить stub
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(self.all_detections, f)

        # Если был один кадр, может вернуть сразу элемент, иначе весь список
        return tracked_list
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

    def draw_bboxes(self, video_frames, results_list):
        output_video_frames = []
        for frame, results in zip(video_frames, results_list):
            annotated = frame.copy()

            # если это Results из ultralytics
            if hasattr(results, 'boxes'):
                # сначала сконвертим его в dict, чтобы не дублировать логику
                tmp = {}
                for box in results.boxes:
                    if results.names[int(box.cls)] == "player":
                        tid = int(box.id.item()) if box.id is not None else None
                        tmp[tid] = box.xyxy[0].tolist()
                player_dict = tmp
            else:
                # уже готовый dict
                player_dict = results

            # рисуем
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(
                    annotated,
                    f"Player ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

            output_video_frames.append(annotated)

        return output_video_frames

    