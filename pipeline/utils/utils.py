import numpy as np

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