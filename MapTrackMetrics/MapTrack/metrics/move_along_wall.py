# import numpy as np
# from shapely.geometry.polygon import Polygon
# from shapely.geometry import Point

# from .metric import Metric
# from .utils import calculate_wallPolygon

# class MoveAlongWall_Metric(Metric):
#     def __init__(self, boundary_region, pWall = 0.15) -> None:
#         super().__init__()
#         self.metricName = "Move Along Wall"
#         self.boundary_region = boundary_region
#         self.pWall = pWall
#         self.wall_polygon = Polygon(calculate_wallPolygon(boundary_region, pWall))
#         self.scores_by_frame = []
    
#     def updateFrame(self, map_pos):
#         super().updateFrame(map_pos)
#         if len(map_pos) > 0:
#             frame_score = 0
#             for _, mapX, mapY in map_pos:
#                 map_point = Point(mapX, mapY)
#                 is_contained = self.wall_polygon.contains(map_point)
#                 if not is_contained:
#                     frame_score += 1
#             self.scores_by_frame.append(frame_score/len(map_pos))

#     def getFinalScore(self) -> float:
#         return round(np.mean(self.scores_by_frame),2)

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from .metric import Metric
from .utils import calculate_wallPolygon

class MoveAlongWall_Metric(Metric):
    def __init__(self, boundary_region, pWall = 0.2) -> None:
        super().__init__()
        self.metricName = "Move Along Wall"
        self.boundary_region = boundary_region
        self.pWall = pWall
        self.wall_polygon = Polygon(calculate_wallPolygon(boundary_region, pWall))
        self.scores_by_frame = {2: [], 3: [], 4: [], 5: []}

    def updateFrame(self, map_pos):
        super().updateFrame(map_pos)
        if len(map_pos) > 0:
            frame_scores = {2: 0, 3: 0, 4: 0, 5: 0}
            person_counts = {2: 0, 3: 0, 4: 0, 5: 0}
            for trk_id, mapX, mapY in map_pos:
                if trk_id in [2, 3, 4, 5]:  # Only consider IDs 2, 3, 4, and 5
                    map_point = Point(mapX, mapY)
                    is_contained = self.wall_polygon.contains(map_point)
                    if not is_contained:
                        frame_scores[trk_id] += 1
                    person_counts[trk_id] += 1

            for trk_id in [2, 3, 4, 5]:
                if person_counts[trk_id] > 0:
                    self.scores_by_frame[trk_id].append(frame_scores[trk_id] / person_counts[trk_id])
                else:
                    self.scores_by_frame[trk_id].append(0)

    def getFinalScore(self) -> dict:
        final_scores = {}
        for trk_id in [2, 3, 4, 5]:
            final_scores[trk_id] = round(np.mean(self.scores_by_frame[trk_id]), 2)
        return final_scores