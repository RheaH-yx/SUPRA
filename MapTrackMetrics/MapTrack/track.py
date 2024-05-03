import numpy as np
import cv2
import imutils
from tqdm import trange
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mf_sort import MF_SORT, Detection
import matplotlib.pyplot as plt
import colors

from mapper import PixelMapper
from metrics import MoveAlongWall_Metric, POD_Metric, EntranceVectors_Metric, TotalEntryTime_Metric, EntranceHesitation_Metric

# video_path = "../Expert/compare_case/10_13_53.mp4" # regular video
# detection_path = "../Expert/compare_case/10_13_53_Detections.txt"
# map_view_path = "../Expert/MapView.jpg"
# point_mapping_path = "../4Man/PointMapping.txt"
# boundary_path = "..//4Man/RoomMapBoundary.txt"
# pod_path = "../4Man/POD.txt"
# output_path = "../Expert/Track.mp4"
# output_track_path = "../Expert/compare_case/10_13_53_Track.txt"

video_path = "/home/rhea/SUPRA/Detection/model/SH_R2_CamF.mp4"
detection_path = "./data/aggregated_results.txt"
map_view_path = "./data/mapView_camF.png"
point_mapping_path = "./data/point_mapping.txt"
boundary_path = "./data/RoomMapBoundary.txt"
pod_path = "./data/POD.txt"
output_path = "./data/output/Track.mp4"
output_track_path = "./data/output/Track.txt"
output_metrics_path = "./data/output/metric_score.txt"


# DISPLAY = False
# RENDER = False
SAVE = True

all_dets = np.loadtxt(detection_path, delimiter=" ")
vs = cv2.VideoCapture(video_path)
map_view_base = cv2.imread(map_view_path)
point_mapping = np.loadtxt(point_mapping_path, delimiter=",", dtype="int")
boundary_region = np.loadtxt(boundary_path, delimiter=",", dtype="int")

metrics = [
    POD_Metric(pod_path),
    MoveAlongWall_Metric(boundary_region),
    EntranceVectors_Metric(),
    TotalEntryTime_Metric(),
    EntranceHesitation_Metric()
]

COLORS = colors.COLORS_THIRTY_NB
mapper = PixelMapper(point_mapping[:,:2], point_mapping[:,2:])
boundary_polygon = Polygon(boundary_region)

mot = MF_SORT()

frame_total = int(all_dets[:,0].max())
all_map_points = []

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
            map_points.append([trk_id, mapX, mapY])
            all_map_points.append([frame_num, trk_id, mapX, mapY])

    for m in metrics:
        m.updateFrame(map_points)

vs.release()

if SAVE:
    res = np.array(all_map_points)
    np.savetxt(output_track_path, res, fmt="%d", delimiter=",")

    with open(output_metrics_path, "w") as f:
        for m in metrics:
            f.write("{}: {}\n".format(m.metricName, m.getFinalScore()))
            print("{}: {}".format(m.metricName, m.getFinalScore()))