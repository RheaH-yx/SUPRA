from functools import cmp_to_key
import numpy as np
from tqdm import tqdm

from .metric import Metric
from .utils import len_comparator, arg_first_comparator, arg_first_non_null

class TotalEntryTime_Metric(Metric):
    def __init__(self, num_tracks = 4) -> None:
        super().__init__()
        self.metricName = "Total Entry Time"
        self.tracks_by_id = {}
        self.frame_count = 0
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

        # Extract the total time based on frame rate
        time_in_frames = frame_ranges[-1][0] - frame_ranges[0][0]
        time_in_seconds = time_in_frames/30.
        # tqdm.write(str(time_in_seconds))
        bottom_clamp = max(0, time_in_seconds - 2)
        score = np.exp(-np.power(bottom_clamp, 2)/4.5)
        return round(score, 2)
