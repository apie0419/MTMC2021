import numpy as np
import os, cv2, math

def getdistance(pt1, pt2):
    EARTH_RADIUS = 6378.137
    lat1, lon1 = pt1[0], pt1[1]
    lat2, lon2 = pt2[0], pt2[1]
    radlat1 = lat1 * math.pi / 180
    radlat2 = lat2 * math.pi / 180
    lat_dis = radlat1 - radlat2
    lon_dis = (lon1 * math.pi - lon2 * math.pi) / 180
    distance = 2 * math.asin(math.sqrt((math.sin(lat_dis/2) ** 2) + math.cos(radlat1) * math.cos(radlat2) * (math.sin(lon_dis/2) ** 2)))
    distance *= EARTH_RADIUS
    distance = round(distance * 10000) / 10000
    return distance

def cosine(vec1, vec2):
    
    num = float(np.matmul(vec1, vec2))
    s = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if s == 0:
        result = 0.0
    else:
        result = num/s
    return result

def get_timestamp_dict(ts_dir):
    ts_dict = dict()
    for filename in os.listdir(ts_dir):
        with open(os.path.join(ts_dir, filename), "r") as f:
            lines = f.readlines()
            temp = dict()
            for line in lines:
                split_line = line.strip("\n").split(" ")
                temp[split_line[0]] = float(split_line[1])
            _max = np.array(list(temp.values())).max()
            for camid, ts in temp.items():
                ts_dict[camid] = ts * -1 + _max

    return ts_dict

def get_fps_dict(work_dir):
    fps_dict = dict()
    for scene_dir in os.listdir(work_dir):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(work_dir, scene_dir)):
                if camera_dir.startswith("c0"):
                    video_path = os.path.join(work_dir, scene_dir, camera_dir, "vdo.avi")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps_dict[camera_dir] = float(fps)
    return fps_dict