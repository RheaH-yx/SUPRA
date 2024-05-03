import json
import cv2
import numpy as np

from mapper import PixelMapper

# scenarioEvents_path = "D:/ARL/data/videos/FtCampbell/4Man/ScenarioEvents.json"
# point_mapping_path = "D:/ARL/data/videos/FtCampbell/4Man/PointMapping.txt"
# map_view_path = "D:/ARL/data/videos/FtCampbell/4Man/MapView.jpg"

scenarioEvents_path = "D:/ARL/data/videos/FtCampbell/4Man/ScenarioEvents.json"
point_mapping_path = "D:/ARL/data/videos/FtCampbell/4Man/PointMapping.txt"
map_view_path = "/home/rhea/SUPRA/Map-SORT/examples/data/SUPRA/mapView_camF.png"

map_view_base = cv2.imread(map_view_path)
point_mapping = np.loadtxt(point_mapping_path, delimiter=",", dtype="int")
mapper = PixelMapper(point_mapping[:,:2], point_mapping[:,2:])

with open(scenarioEvents_path, "r") as ef:
    scenarioData = json.load(ef)

for character in scenarioData["characters"]:
    pos = character["position"]
    mapCoord = mapper.pixel_to_map(pos).ravel()
    mapX, mapY = int(mapCoord[0]), int(mapCoord[1])
    print(pos, ":", (mapX, mapY))
    cv2.circle(map_view_base, (mapX, mapY), 5, (0,255,0), -1)

cv2.imshow("Map", map_view_base)
cv2.waitKey(0)