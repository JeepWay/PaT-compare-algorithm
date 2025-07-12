import numpy as np
from shelf_fit.common.constants import BIN, MASK
import logging


class Rectangle:
    def __init__(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        self.x = x  # 寬度的座標
        self.y = y  # 高度的座標


class Shelf:
    def __init__(self, y, height, bin_width):
        self.y = y                      # 起始高度的座標
        self.height = height            # 貨架高度
        self.bin_width = bin_width      # 貨架最大寬度
        self.current_x = 0              # 目前放置到哪一個寬度座標
        self.rectangles: list[Rectangle] = []   # 紀錄放置的矩形

    def can_fit(self, rect: Rectangle):
        return (self.current_x + rect.width <= self.bin_width and 
                rect.height <= self.height)

    def add_rectangle(self, rect: Rectangle):
        rect.x = self.current_x
        rect.y = self.y
        self.rectangles.append(rect)
        self.current_x += rect.width


class ShelfFit:
    def __init__(self, bin_w, bin_h):
        self.bin_width = bin_w
        self.bin_height = bin_h
        self.reset()

    def reset(self):
        self.next_y = 0
        self.shelves: list[Shelf] = []
        self.used_area = 0
        self.current_shelf = None  # 用於 NextFit

    def get_shelves_info(self):
        """
        回傳所有 shelf 的資訊。
        格式：[ {y, height, rectangles: [(x, y, w, h), ...]}, ... ]
        """
        return [
            {
                "shelf_y": shelf.y,
                "shelf_height": shelf.height,
                "rectangles": [(r.x, r.y, r.width, r.height) for r in shelf.rectangles]
            }
            for shelf in self.shelves
        ]

    def parse_observation(self, observation):
        bin_data = observation[BIN][0][0]  # shape: (width, height)
        width = int(observation[BIN][0, 1, 0, 0])
        height = int(observation[BIN][0, 2, 0, 0])
        logging.info(f"Bin data: \n{bin_data}")
        logging.info(f"Item size: {width}x{height}")
        return Rectangle(width, height)

    def place_and_return_action(self, rect: Rectangle, shelf: Shelf, state):
        shelf.add_rectangle(rect)
        self.used_area += rect.width * rect.height
        action = (self.bin_width * rect.x) + rect.y
        logging.debug(f"放置位置: (x={rect.x}, y={rect.y}, action={action})")
        return np.array([action], dtype=np.int64), state

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        raise NotImplementedError("Subclasses must implement predict()")


class ShelfNextFit(ShelfFit):
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        rect = self.parse_observation(observation)

        if self.current_shelf is None:
            logging.debug("沒有任何 shelf，創建新的 shelf")
            self.current_shelf = Shelf(self.next_y, rect.height, self.bin_width)
            self.shelves.append(self.current_shelf)
            self.next_y += rect.height
        elif not self.current_shelf.can_fit(rect):
            if self.next_y + rect.height > self.bin_height:
                logging.debug("沒有足夠的高度開啟新的 shelf，返回動作 -1")
                return np.array([-1], dtype=np.int64), state
            logging.debug(f"開啟新的 shelf, 起始位置: {self.next_y}")
            self.current_shelf = Shelf(self.next_y, rect.height, self.bin_width)
            self.shelves.append(self.current_shelf)
            self.next_y += rect.height

        return self.place_and_return_action(rect, self.current_shelf, state)


class ShelfFirstFit(ShelfFit):
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        rect = self.parse_observation(observation)

        for shelf in self.shelves:
            if shelf.can_fit(rect):
                return self.place_and_return_action(rect, shelf, state)

        # 沒有任何 shelf 可放，嘗試開啟新的 shelf
        if self.next_y + rect.height > self.bin_height:
            logging.debug("沒有足夠的高度開啟新的 shelf，返回動作 -1")
            return np.array([-1], dtype=np.int64), state

        logging.debug(f"開啟新的 shelf, 起始位置: {self.next_y}")
        new_shelf = Shelf(self.next_y, rect.height, self.bin_width)
        self.shelves.append(new_shelf)
        self.next_y += rect.height

        return self.place_and_return_action(rect, new_shelf, state)
