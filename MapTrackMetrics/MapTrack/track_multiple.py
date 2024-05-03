import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm, trange
from mf_sort import MF_SORT, Detection

from metrics import *
import colors
from mapper import PixelMapper


# base_dir = "D:/ARL/data/videos/FtCampbell/Cam3/9-8-21/Part_2"
# point_mapping_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-8-21/Part_2/PointMapping.txt"
# boundary_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-8-21/Part_2/RoomMapBoundary.txt"
# pod_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-8-21/Part_2/POD.txt"

base_dir = "./data"
point_mapping_path = "./data/point_mapping.txt"
boundary_path = "./data/RoomMapBoundary.txt"
# pod_path = "D:/ARL/data/videos/FtCampbell/Cam3/9-8-21/Part_2/POD.txt"

point_mapping = np.loadtxt(point_mapping_path, delimiter=",", dtype="int")
boundary_region = np.loadtxt(boundary_path, delimiter=",", dtype="int")
COLORS = colors.COLORS_THIRTY_NB
detections_dir = os.path.join(base_dir, "Detections")

mapper = PixelMapper(point_mapping[:,:2], point_mapping[:,2:])
boundary_polygon = Polygon(boundary_region)

metrics_out = []
for filename in tqdm(os.listdir(detections_dir), position=0, desc="Videos"):
    detection_path = os.path.join(detections_dir, filename)
    all_dets = np.loadtxt(detection_path, delimiter=",")
    frame_total = int(all_dets[:,0].max())
    mot = MF_SORT()

    #tqdm.write(filename)

    metrics = [
        POD_Metric(pod_path, 3),
        MoveAlongWall_Metric(boundary_region),
        EntranceVectors_Metric(num_tracks=3),
        TotalEntryTime_Metric(num_tracks=3),
        EntranceHesitation_Metric(num_tracks=3)
    ] 

    for frame_num in trange(frame_total, position=1, desc="Frames", leave=False):
        dets = all_dets[(all_dets[:,0] == frame_num) & (all_dets[:,1] == 1) & (all_dets[:,-1] >= 0.5)]
        dets = [Detection(det[2:6], det[6]) for det in dets]
        trks = mot.step(dets)

        map_points = []
        for trk, trk_id in trks:
            x, y, w, h = [int(i) for i in trk.tlwh]
            bottomCenter = x+(w/2), y+h
            mapCoord = mapper.pixel_to_map(bottomCenter).ravel()
            mapX, mapY = int(mapCoord[0]), int(mapCoord[1])

            map_point = Point(mapX, mapY)
            if boundary_polygon.contains(map_point):
                map_points.append([trk_id, mapX, mapY])
        
        for m in metrics:
            m.updateFrame(map_points)
    
    metrics_out.append([filename] + [m.getFinalScore() for m in metrics])

out_path = os.path.join(base_dir, "Metrics.csv")
out_header = ["Filename"] + [m.metricName for m in metrics]
res = pd.DataFrame(metrics_out, columns=out_header)
res.to_csv(out_path, float_format="%.2f", index=False)