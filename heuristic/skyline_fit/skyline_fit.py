import numpy as np
from shelf_fit.common.constants import BIN, MASK
import json
import logging


class Rectangle:
    def __init__(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        self.x = x  # 寬度的座標
        self.y = y  # 高度的座標


class SkylineNode:
    def __init__(self, x, y, width):
        self.x = x  # 起始寬度座標
        self.y = y  # 高度
        self.width = width


class SkylineBottomLeft:
    def __init__(self, bin_w, bin_h):
        self.bin_width = bin_w     # width (x-axis)
        self.bin_height = bin_h    # height (y-axis)
        self.skyline: list[SkylineNode] = []
        self.rectangles: list[Rectangle] = []
        self.reset()

    def reset(self):
        self.skyline : list[SkylineNode] = [SkylineNode(0, 0, self.bin_width)]
        self.rectangles = []
        self.used_surface_area = 0

    def can_place(self, bin_data, x, y, w, h):
        if x + w > self.bin_width or y + h > self.bin_height:
            logging.debug(f"Can not fit [exceed bin size]: (x, y, w, h) = ({x}, {y}, {w}, {h})")
            return False
        
        region = bin_data[x : x+w, y : y+h]
        can_place = np.all(region == 1)
        
        if not can_place:
            logging.debug(f"Can not fit [occupied]: (x, y, w, h) = ({x}, {y}, {w}, {h}), region:{region}")
        else: 
            logging.debug(f"Can fit: (x, y, w, h) = ({x}, {y}, {w}, {h}), region:{region}")
        return can_place

    def add_skyline_level(self, index, x, y, width, height):
        new_node = SkylineNode(x, y + height, width)
        self.skyline.insert(index, new_node)

        i = index + 1
        while i < len(self.skyline):
            node = self.skyline[i]
            prev = self.skyline[i - 1]
            if node.x < prev.x + prev.width:
                shrink = prev.x + prev.width - node.x
                node.x += shrink
                node.width -= shrink
                if node.width <= 0:
                    self.skyline.pop(i)
                    continue
            else:
                break
            i += 1

        self.merge_skyline()

    def merge_skyline(self):
        i = 0
        while i < len(self.skyline) - 1:
            if self.skyline[i].y == self.skyline[i + 1].y:
                self.skyline[i].width += self.skyline[i + 1].width
                self.skyline.pop(i + 1)
            else:
                i += 1

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        bin_data = observation[BIN][0][0]  # shape = (width, height)
        width = int(observation[BIN][0, 1, 0, 0])
        height = int(observation[BIN][0, 2, 0, 0])
        logging.info(f"Bin data: \n{bin_data}")
        logging.info(f"Item size: {width}x{height}")
        
        rect = Rectangle(width, height)
        
        best_x = best_y = None
        best_index = -1         # 最佳的天際線
        best_y_score = None       # 最低高度

        for i, node in enumerate(self.skyline):
            x = node.x
            y = node.y
            if not self.can_place(bin_data, x, y, rect.width, rect.height):
                continue
            if best_y_score is None or (y < best_y_score) or (y == best_y_score and x < best_x):
                logging.debug("新的最佳位置")
                best_x = x
                best_y = y
                best_index = i
                best_y_score = y

        if best_index == -1:
            logging.debug("無法放置，返回動作 -1")
            return np.array([-1], dtype=np.int64), state

        self.add_skyline_level(best_index, best_x, best_y, rect.width, rect.height)
        self.used_surface_area += (rect.width * rect.height)

        rect.x = best_x
        rect.y = best_y
        self.rectangles.append(rect)
        
        action = best_x * self.bin_height + best_y
        logging.debug(f"放置位置: (x={rect.x}, y={rect.y}, action={action})")
        return np.array([action], dtype=np.int64), state
