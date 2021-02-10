import math

class Track(object):

    def __init__(self):
        self.id = None
        self.gps_list = list()
        self.feature_list = list()
        self.frame_list = list()
        self.box_list = list()
        self.ts_list = list()
    
    def __len__(self):
        return len(self.frame_list)

    def speed(self):
        pt1 = self.gps_list[0]
        pt2 = self.gps_list[-1]
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
        t = self.ts_list[-1] - self.ts_list[0]
        if t ==0:
            return 0
        return distance / t