import numpy as np
import cv2


class PixelMapper(object):
    def __init__(self, pixel_array, target_array):
        pixel_array = np.asarray(pixel_array)
        target_array = np.asarray(target_array)

        assert pixel_array.shape == (4, 2), "Need (4,2) input array"
        assert target_array.shape == (4, 2), "Need (4,2) input array"

        self.M = cv2.getPerspectiveTransform(
            np.float32(pixel_array), np.float32(target_array))
        self.invM = cv2.getPerspectiveTransform(
            np.float32(target_array), np.float32(pixel_array))

    def pixel_to_map(self, pixel):
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1, 2)
        assert pixel.shape[1] == 2, "Need (N,2) input array"

        pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
        target = np.dot(self.M, pixel.T)

        return (target[:2, :]/target[2, :]).T

    def map_to_pixel(self, coord):
        if type(coord) != np.ndarray:
            coord = np.array(coord).reshape(1, 2)
        assert coord.shape[1] == 2, "Need (N,2) input array"

        coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis=1)
        pixel = np.dot(self.invM, coord.T)

        return (pixel[:2, :]/pixel[2, :]).T