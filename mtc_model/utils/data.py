import torch, random

from tqdm import tqdm

class Dataset(object):

    def __init__(self, filename):
        self.data_dict = self.read_tracklets_file(filename)

    def __len__(self):
        count = 0
        for camera_id in self.data_dict:
            for det_id in self.data_dict[camera_id]:
                for camid in self.data_dict:
                    if camid == camera_id:
                        continue
                    cam_data = self.data_dict[camera_id]
                    if det_id not in cam_data:
                        continue
                    count += 1
        return count

    def read_tracklets_file(self, filename):
        data_dict = dict()
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading Tracklets File"):
                words = line.strip("\n").split(',')
                camera_id = words[0]
                frame_id = words[1]
                det_id = words[2]
                features = list(map(float, words[3:]))
                if camera_id not in data_dict:
                    data_dict[camera_id] = dict()
                if det_id not in data_dict[camera_id]:
                    data_dict[camera_id][det_id] = list()

                data_dict[camera_id][det_id].append(features)
        
        return data_dict

    def prepare_data(self):
        count = 0
        for camera_id in self.data_dict:
            for det_id in self.data_dict[camera_id]:
                query_track = self.data_dict[camera_id][det_id]
                query_track = torch.tensor(query_track)
                mean = query_track.mean(dim=0)
                std = query_track.std(dim=0)
                query = torch.cat((mean, std))
                if True in torch.isnan(query).numpy().tolist():
                    continue

                for camid in self.data_dict:
                    tracklets = [query]
                    gallery_tracks = list()
                    if camid == camera_id:
                        continue
                    cam_data = self.data_dict[camid]

                    if det_id in cam_data:
                        gallery_track = torch.tensor(cam_data[det_id])
                        mean = gallery_track.mean(dim=0)
                        std  = gallery_track.std(dim=0)
                        gallery = torch.cat((mean, std))
                        if True in torch.isnan(gallery).numpy().tolist():
                            continue
                        gallery_tracks.append(gallery)
                        
                    else:
                        continue

                    _min = 10
                    _max = 15
                    if len(cam_data) - 1 < _max:
                        _max = len(cam_data) - 1
                        if len(cam_data) - 1 < _min:
                            _min = len(cam_data) - 1

                    num_objects = random.randint(_min, _max)
                    ids = list(cam_data.keys())
                    random.shuffle(ids)

                    for id in ids:
                        if (id == det_id) or (num_objects == 0):
                            continue
                        gallery_track = torch.tensor(cam_data[id])
                        mean = gallery_track.mean(dim=0)
                        std  = gallery_track.std(dim=0)
                        gallery = torch.cat((mean, std))
                        if True in torch.isnan(gallery).numpy().tolist():
                            continue
                        gallery_tracks.append(gallery)
                        num_objects -= 1
                    
                    if len(gallery_tracks) == 1:
                        continue

                    orders = list(range(len(gallery_tracks)))
                    tmp = list(zip(orders, gallery_tracks))
                    random.shuffle(tmp)
                    orders, gallery_tracks = zip(*tmp)
                    label = torch.tensor([orders.index(0)]).long()
                    tracklets.extend(gallery_tracks)
                    data = torch.stack(tracklets)
                    yield data, label