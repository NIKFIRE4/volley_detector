import cv2
import numpy as np
import constants  # здесь твои волейбольные константы
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self, frame):
        # размеры фонового прямоугольника
        self.drawing_rectangle_width = 150
        self.drawing_rectangle_height = 300
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def convert_meters_to_pixels(self, meters):
        # reference: полная ширина площадки constants.FACE_LINE_WIDTH
        return convert_meters_to_pixel_distance(
            meters,
            constants.FACE_LINE_WIDTH,
            self.court_drawing_width
        )

    def set_court_drawing_key_points(self):
        dk = [0] * 24

        # 0–3: углы корта (по часовой)
        dk[0], dk[1] = self.court_start_x, self.court_start_y
        dk[2], dk[3] = self.court_end_x,   self.court_start_y
        dk[4], dk[5] = self.court_end_x,   self.court_end_y
        dk[6], dk[7] = self.court_start_x, self.court_end_y

        # 8–11: сетка по центру
        net_y = self.court_start_y + self.convert_meters_to_pixels(constants.LENGTH_BEFORE_GRID)
        dk[8],  dk[9]  = self.court_start_x, int(net_y)
        dk[10], dk[11] = self.court_end_x,   int(net_y)

        # 12–19: линии атаки (3 м от сетки)
        ao = self.convert_meters_to_pixels(constants.LENGTH_ATTACK_BEFORE_AVERAGE)
        dk[12], dk[13] = self.court_start_x, int(net_y - ao)
        dk[14], dk[15] = self.court_end_x,   int(net_y - ao)
        dk[16], dk[17] = self.court_start_x, int(net_y + ao)
        dk[18], dk[19] = self.court_end_x,   int(net_y + ao)

        # 20–23: центральная линия (по X) — если нужна
        # mid_x = int((self.court_start_x + self.court_end_x) / 2)
        # dk[20], dk[21] = mid_x, self.court_start_y
        # dk[22], dk[23] = mid_x, self.court_end_y

        self.drawing_key_points = dk

    def set_court_lines(self):
    # 0–3: контур площадки
    # 4–5: сетка
    # 6–7: линия атаки в верхней половине
    # 8–9: линия атаки в нижней половине
    # 10–11: центральная вертикальная линия (опционально)
        self.lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # контур корта
            (4, 5),                          # сетка
            (6, 7),                          # линия атаки ближе к start_y
            (8, 9),                          # линия атаки ближе к end_y
            # (10, 11)                       # средняя вертикаль, если нужна
        ]

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes,
                      (self.start_x, self.start_y),
                      (self.end_x,   self.end_y),
                      (255,255,255), cv2.FILLED)
        alpha = 0.5
        out = frame.copy()
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1-alpha, 0)[mask]
        return out

    def draw_court(self, frame):
        # точки
        for i in range(0, len(self.drawing_key_points), 2):
            x, y = int(self.drawing_key_points[i]), int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y), 4, (0,0,255), -1)

        # линии
        for a, b in self.lines:
            p1 = (int(self.drawing_key_points[2*a]),   int(self.drawing_key_points[2*a+1]))
            p2 = (int(self.drawing_key_points[2*b]),   int(self.drawing_key_points[2*b+1]))
            cv2.line(frame, p1, p2, (0,0,0), 2)

        return frame

    def draw_mini_court(self, frames):
        out = []
        for f in frames:
            f_bg = self.draw_background_rectangle(f)
            f_court = self.draw_court(f_bg)
            out.append(f_court)
        return out
