from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.all_detections = {}
    def track(self, frames, frame_idx, show=False, tracker=None,
              read_from_stub=False, stub_path=None):
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

    

    def get_ball_shot_frames(self, ball_positions):
        rows = []
        if not ball_positions:
            return []
            
        # Собираем данные по всем кадрам
        for frame_idx, res in ball_positions.items():
            if res is None:
                rows.append({'frame_idx': frame_idx, 'x1': None, 'y1': None, 'x2': None, 'y2': None})
                continue
                
            boxes = res.boxes
            if boxes and len(boxes.xyxy) > 0:
                coords = boxes.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = coords
            else:
                x1 = y1 = x2 = y2 = None

            rows.append({
                'frame_idx': frame_idx,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

        # Создаем DataFrame и заполняем пропуски
        df_ball_positions = pd.DataFrame(rows).set_index('frame_idx')
        
        # Реиндексируем чтобы заполнить пропущенные кадры
        max_frame = df_ball_positions.index.max() if not df_ball_positions.empty else 0
        df_ball_positions = df_ball_positions.reindex(range(max_frame + 1))
        
        # Интерполируем только числовые колонки
        numeric_cols = ['x1', 'y1', 'x2', 'y2']
        df_ball_positions[numeric_cols] = df_ball_positions[numeric_cols].interpolate().ffill().bfill()
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()
        df_ball_positions['ball_hit'] = 0

        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        return frame_nums_with_ball_hits



    def draw_bboxes(self, frames, tracked_list, color=(0,0,255), thickness=2):
        """
        frames:      список numpy.ndarray (входные кадры)
        tracked_list: список ultralytics.engine.results.Results (результаты трекинга)
        """
        output = []

        for frame, res in zip(frames, tracked_list):
            img = frame.copy()
            
            # Проверяем, есть ли боксы
            if res.boxes is not None and len(res.boxes.xyxy) > 0:
                # Преобразуем тензор в numpy array
                bboxes = res.boxes.xyxy.cpu().numpy()
                for x1, y1, x2, y2 in bboxes:
                    # Рисуем прямоугольник
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness
                    )
            # Если боксов нет, оставляем кадр без изменений
            output.append(img)

        return output
