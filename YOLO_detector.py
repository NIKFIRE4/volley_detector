import torch
import numpy as np
from ultralytics import YOLO
import os
import cv2
import easyocr
import matplotlib.pyplot as plt
class ObjectTracking:
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(dir, r"best2.pt")
        self.video_path = os.path.join(dir, r"volley_detector\training\videos_volley\7 - Trim.mp4")
        self.bytetrack_yaml_path = os.path.join(dir, "bytetrack.yaml")
        self.model = YOLO(weights_path)
        self.reader = easyocr.Reader(['ru','en'])

    def number_recognition(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]  # Вырезаем область номера

        # Отобразим изображение для отладки
        # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()

        # Распознаём текст
        result = self.reader.readtext(roi, detail=0)
        
        return result
    
    def detect_object(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
        
        results = self.model.predict(source=self.video_path, stream=True, line_width=1)
        for result in results:
            frame = result.plot()  # Получаем кадр с нарисованными bounding boxes
            out.write(frame)
        
        cap.release()
        out.release()
    def tracking_object(self, numbers_players):
        """Отслеживает только игроков с нужными номерами."""
        cap = cv2.VideoCapture(self.video_path)
        selected_players = {}  # Храним соответствие ID ↔ номер игрока

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(source=frame, persist=True, tracker=self.bytetrack_yaml_path)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
            ids = results[0].boxes.id.cpu().numpy().astype(int)  # Tracker IDs

            for box, id in zip(boxes, ids):
                if id not in selected_players:  
                    detected_numbers = self.number_recognition(frame, box)  # Распознаем номер
                    if any(num in numbers_players for num in detected_numbers):  # Проверяем, есть ли номер в списке
                        selected_players[id] = detected_numbers[0]  # Запоминаем ID игрока

                if id in selected_players:  # Отрисовываем только выбранных игроков
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                    cv2.putText(frame, f"ID {id} - {selected_players[id]}", 
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()    
    
            
            
def run_detect_object():
    ot = ObjectTracking()
    ot.detect_object()
def run_track_object():
    ot = ObjectTracking()
    ot.tracking_object(numbers_players=["5", "10", "15"])

    

if __name__ == "__main__":
    run_track_object()


