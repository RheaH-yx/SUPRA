from operator import itemgetter
import numpy as np
import cv2
import imutils
from sqlalchemy import true
from tqdm import trange, tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mf_sort import MF_SORT, Detection
import matplotlib.pyplot as plt
import colors
from collections import defaultdict
import os

from mapper import PixelMapper

# video_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-9-21/Videos/10_23_22.mp4"
# detection_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-9-21/Detections/10_23_22.txt"

# map_view_path = "D:/ARL/data/videos/FtCampbell/4Man/MapView.jpg"
# point_mapping_path = "D:/ARL/data/videos/FtCampbell/4Man/PointMapping.txt"
# boundary_path = "D:/ARL/data/videos/FtCampbell/4Man/RoomMapBoundary.txt"
# pod_path = "D:/ARL/data/videos/FtCampbell/4Man/POD.txt"
# output_path = ""


video_path = "/home/rhea/SUPRA/Detection/model/SH_R2_CamF.mp4"
detection_path = "./data/aggregated_results.txt"
map_view_path = "./data/mapView_camF.png"
point_mapping_path = "./data/point_mapping.txt"
boundary_path = "./data/RoomMapBoundary.txt"
pod_path = "./data/POD.txt"
output_path = ""

all_dets = np.loadtxt(detection_path, delimiter=" ")
vs = cv2.VideoCapture(video_path)
map_view_base = cv2.imread(map_view_path)
point_mapping = np.loadtxt(point_mapping_path, delimiter=",", dtype="int")
boundary_region = np.loadtxt(boundary_path, delimiter=",", dtype="int")

COLORS = colors.COLORS_THIRTY_NB
mapper = PixelMapper(point_mapping[:,:2], point_mapping[:,2:])
boundary_polygon = Polygon(boundary_region)
mot = MF_SORT()
frame_total = int(all_dets[:,0].max())
map_track_points = defaultdict(lambda: [])

for frame_num in trange(frame_total):
    ret, frame = vs.read()
    if frame is None or not ret:
        break

    dets = all_dets[(all_dets[:,0] == frame_num) & (np.isin(all_dets[:,1], [2, 3, 4, 5])) & (all_dets[:,-1] >= 0.5)]
    dets = [Detection(det[2:6], det[6]) for det in dets]
    trks = mot.step(dets)

    map_points = []
    for trk, trk_id in trks:
        x, y, w, h = [int(i) for i in trk.tlwh]
        bottomCenter = x, y+h/2
        mapCoord = mapper.pixel_to_map(bottomCenter).ravel()
        mapX, mapY = int(mapCoord[0]), int(mapCoord[1])

        map_point = Point(mapX, mapY)
        if boundary_polygon.contains(map_point):
            map_track_points[trk_id].append((mapX, mapY))

if len(map_track_points.keys()) > 4:
    lengths = []
    for trk_id, points in map_track_points.items():
        lengths.append((trk_id, len(points)))
    lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
    for idx, _ in lengths[4:]:
        map_track_points.pop(idx)


map_frame = map_view_base.copy()
j = 0
for trk_id, points in tqdm(map_track_points.items()):
    color = COLORS[j]
    j += 1
    cv2.circle(map_frame, points[0], 3, color, thickness=-1)

    for i in range(1,len(points)):
        radius = 3
        if i == (len(points)-1):
            radius = 10
        
        point1 = points[i-1]
        point2 = points[i]
        cv2.line(map_frame, point1, point2, color, 2)
        cv2.circle(map_frame, point2, radius, color, thickness=-1)

outfile = os.path.join("motion_paths",os.path.splitext(os.path.basename(video_path))[0] + ".png")
cv2.imwrite(outfile, map_frame)

plt.figure()
plt.imshow(cv2.cvtColor(map_frame, cv2.COLOR_BGR2RGB))
plt.title("High Trainee Performance Example")
plt.axis("off")
plt.show()