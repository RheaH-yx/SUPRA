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
    def __init__(self, pod_file) -> None:
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
        tracks = tracks[:4]
        tracks.sort(key=cmp_to_key(arg_first_comparator))
        frame_ranges = [(arg_first_non_null(trk), arg_first_non_null(reversed(trk))) for trk in tracks]

        stop_points = []
        for i, trk in enumerate(tracks):
            # Interpolate missing frames in the track and savgol smooth to reduce noise
            start_frame, end_frame = frame_ranges[i]
            end_point = -end_frame if end_frame != 0 else len(trk)
            df = pd.DataFrame([(None, None) if v is None else v for v in trk[start_frame:end_point]])
            trk_interp = df.interpolate().to_numpy()
            trk_interp[:,0] = signal.savgol_filter(trk_interp[:,0], 11, 2)
            trk_interp[:,1] = signal.savgol_filter(trk_interp[:,1], 11, 2)

            # Calculate degree of motion curve based on 5th sum of successive euclid distances
            diffs = np.diff(trk_interp, axis=0)
            dists = np.hypot(diffs[:,0], diffs[:,1])
            sum_dists = np.add.reduceat(dists, np.arange(0, len(diffs), 15))
            
            # Smooth the motion curve agressively, then find knee of the curve to find
            # the first point where soldiers stop
            window_length = int(len(sum_dists)/4)
            if window_length%2 == 0:
                window_length -= 1
            sum_dists = signal.savgol_filter(sum_dists, window_length, 1)
            for settings in self.curve_settings:
                kn = KneeLocator(range(len(sum_dists)), sum_dists, curve=settings[0], direction=settings[1])
                if kn.knee != 0:
                    break
            stop_points.append(trk_interp[kn.knee*15,:])

        # Minimum cost assignment of stopping points to the expected PODs
        cost_matrix = np.zeros((4,4))
        for i, sp in enumerate(stop_points):
            for j, pod_point in enumerate(self.pod):
                cost_matrix[i,j] = distance.euclidean(sp, pod_point)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        # Use average of guassians to detemine final score
        scores = []
        for row, col in zip(row_idx, col_idx):
            scores.append(np.exp(-np.power(cost_matrix[row, col], 2)/7200))
        return round(np.mean(scores), 2)