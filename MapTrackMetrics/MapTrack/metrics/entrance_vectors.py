from functools import cmp_to_key
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from kneed import KneeLocator
from tqdm.std import tqdm

from .metric import Metric
from .utils import get_odd, len_comparator, arg_first_comparator, arg_first_non_null

class EntranceVectors_Metric(Metric):
    def __init__(self, num_tracks = 4) -> None:
        super().__init__()
        self.metricName = "Entrance Vectors"
        self.tracks_by_id = {}
        self.frame_count = 0
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

        if len(tracks) < 2:
            return -1

        entrance_angles = []
        for i, trk in enumerate(tracks):
            # Interpolate missing frames in the track
            start_frame, end_frame = frame_ranges[i]
            end_point = -end_frame if end_frame != 0 else len(trk)
            df = pd.DataFrame([(None, None) if v is None else v for v in trk[start_frame:end_point]])
            trk_interp = df.interpolate().to_numpy()

            window_size = min(len(trk_interp), 11)
            if window_size%2 == 0:
                window_size -= 1
            trk_interp[:,0] = savgol_filter(trk_interp[:,0], window_size, 2)
            trk_interp[:,1] = savgol_filter(trk_interp[:,1], window_size, 2)

            # Calculate degree of motion curve based on 5th sum of successive euclid distances
            diffs = np.diff(trk_interp, axis=0)
            dists = np.hypot(diffs[:,0], diffs[:,1])
            #sum_dists = np.add.reduceat(dists, np.arange(0, len(diffs), 5))
            sum_dists = np.convolve(dists, np.ones(15), "valid")/15
            
            try:
                # Smooth the motion curve agressively, then find knee of the curve to find
                # the first point where soldiers stop
                window_length = get_odd(int(len(sum_dists)/4))
                sum_dists = savgol_filter(sum_dists, window_length, 1)
                hist, bin_edges = np.histogram(sum_dists)
                sum_dists[sum_dists <= bin_edges[1]] = 0
                mask = sum_dists <= bin_edges[1]
                stop_idx = int(np.where(mask.any(), mask.argmax(), -1))
                if stop_idx == -1:
                    raise ValueError

                # Calculate soldier's entrance vector and angle
                start_point = np.array(trk_interp[0,:])
                end_point = np.array(trk_interp[stop_idx, :])
                entrance_vector = end_point - start_point
                entrance_unit = entrance_vector / np.linalg.norm(entrance_vector)
                angle = np.arccos(np.dot(entrance_unit, np.array([1,0])))
                entrance_angles.append(np.pi/2 - angle)
            except:
                entrance_angles.append(-1 * entrance_angles[-1])

        # Calculate final scores based on percentage of correct
        score = 0
        for i, angle_diff in enumerate(entrance_angles[1:]):
            if np.sign(angle_diff) != np.sign(entrance_angles[i]):
                score += 1
        return round(score / 3., 2)
 