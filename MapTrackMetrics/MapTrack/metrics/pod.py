import numpy as np
from scipy.spatial import distance
from scipy import signal
from functools import cmp_to_key
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from .metric import Metric
from .utils import len_comparator, arg_first_comparator, arg_first_non_null

class POD_Metric(Metric):
    def __init__(self, pod_file, num_tracks = 4) -> None:
        super().__init__()
        self.metricName = "Points of Domination"
        self.tracks_by_id = {}
        self.frame_count = 0
        self.pod = np.loadtxt(pod_file, delimiter=",", dtype="int")
        self.curve_settings = [
            ("convex", "decreasing"), 
            ("concave", "increasing"), 
            ("concave", "decreasing"), 
            ("convex", "increasing")
        ]
        self.num_tracks = num_tracks

    def updateFrame(self, map_pos):
        super().updateFrame(map_pos)
        self.frame_count += 1

        processed_ids = []
        if len(map_pos) > 0:
            for idx, mapX, mapY in map_pos:
                if idx not in self.tracks_by_id:
                    self.tracks_by_id[idx] = [None for i in range(self.frame_count-1)]
                self.tracks_by_id[idx].append((mapX, mapY))
                processed_ids.append(idx)
        
        for idx in self.tracks_by_id.keys():
            if idx not in processed_ids:
                self.tracks_by_id[idx].append(None)

    def getFinalScore(self) -> float:
        # Get the 4 longest tracks and sort them by their start frames
        tracks = list(self.tracks_by_id.values())
        tracks.sort(key=cmp_to_key(len_comparator), reverse=True)
        tracks = tracks[:self.num_tracks]
        tracks.sort(key=cmp_to_key(arg_first_comparator))
        frame_ranges = [(arg_first_non_null(trk), arg_first_non_null(reversed(trk))) for trk in tracks]

        if len(tracks) < self.num_tracks:
            return -1

        all_dists = []
        for i, trk in enumerate(tracks):
            # Interpolate missing frames in the track and savgol smooth to reduce noise
            start_frame, end_frame = frame_ranges[i]
            end_point = -end_frame if end_frame != 0 else len(trk)
            df = pd.DataFrame([(None, None) if v is None else v for v in trk[start_frame:end_point]])
            trk_interp = df.interpolate().to_numpy()
            
            window_size = min(len(trk_interp), 11)
            if window_size%2 == 0:
                window_size -= 1
            trk_interp[:,0] = signal.savgol_filter(trk_interp[:,0], window_size, 2)
            trk_interp[:,1] = signal.savgol_filter(trk_interp[:,1], window_size, 2)

            dists = []
            for pod_point in self.pod:
                d = np.linalg.norm(trk_interp - pod_point, axis=1)
                dists.append((np.mean(d), d.min()))
            all_dists.append(dists)
        
        # Minimum cost assignment to the expected PODs
        cost_matrix = np.zeros((self.num_tracks,len(self.pod)))
        for soldier_num in range(self.num_tracks):
            for pod_num in range(len(self.pod)):
                cost_matrix[soldier_num, pod_num] = all_dists[soldier_num][pod_num][0]
        soldier_idx, pod_idx = linear_sum_assignment(cost_matrix)

        scores = []
        for soldier_num, pod_num in zip(soldier_idx, pod_idx):
            scores.append(np.exp(-np.power(all_dists[soldier_num][pod_num][1], 2)/5000))
        return round(np.mean(scores), 2)