import cv2

from colors import COLORS_THIRTY_NB

class DrawLineWidget(object):
    def __init__(self, img_path):
        self.original_image = cv2.imread(img_path)
        self.clone = self.original_image.copy()
        self.colors = COLORS_THIRTY_NB
        self.color_idx = 0

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates.append((x,y))
            color = self.colors[self.color_idx]
            cv2.circle(self.clone, self.image_coordinates[-1], 3, color, thickness=-1)

            if(len(self.image_coordinates) > 1):
                point1 = self.image_coordinates[-2]
                point2 = self.image_coordinates[-1]
                cv2.line(self.clone, point1, point2, color, 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            color = self.colors[self.color_idx]
            cv2.circle(self.clone, self.image_coordinates[-1], 10, color, thickness=-1)
            self.color_idx += 1
            self.image_coordinates = []

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    map_view_path = "D:/ARL/data/videos/FtCampbell/4Man/MapView.jpg"
    out_path = "motion_paths/expert.png"
    draw_line_widget = DrawLineWidget(map_view_path)

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.imwrite(out_path, draw_line_widget.show_image())
            cv2.destroyAllWindows()
            exit(1)