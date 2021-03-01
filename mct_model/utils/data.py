import torch, random

from tqdm import tqdm

class Dataset(object):

    def __init__(self, feature_file, easy_tracklets_file, hard_tracklets_file):
        self.feature_dict = self.read_feature_file(feature_file)
        self.easy_data_list = self.read_tracklets_file(easy_tracklets_file)
        self.hard_data_list = self.read_tracklets_file(hard_tracklets_file)

    def hard_len(self):
        return len(self.hard_data_list)

    def easy_len(self):
        return len(self.easy_data_list)

    def read_feature_file(self, filename):
        feature_dict = dict()
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading Feature File"):
                words = line.strip("\n").split(',')
                camera_id = words[0]
                frame_id = words[1]
                det_id = words[2]
                features = list(map(float, words[3:]))
                if camera_id not in feature_dict:
                    feature_dict[camera_id] = dict()
                if det_id not in feature_dict[camera_id]:
                    feature_dict[camera_id][det_id] = list()

                feature_dict[camera_id][det_id].append(features)
        
        return feature_dict

    def read_tracklets_file(self, filename):
        data_list = list()
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading Tracklets File"):
                words = line.strip("\n").split(' ')
                q_cam = words[0]
                g_cam = words[1]
                q_id = words[2]
                g_ids = words[3].split(',')
                label = int(words[4])
                data_list.append([q_cam, g_cam, q_id, g_ids, label])
        
        return data_list

    def prepare_data(self, _type):
        
        if _type == "easy":
            data_list = self.easy_data_list
        elif _type == "hard":
            data_list = self.hard_data_list
        elif _type == "merge":
            data_list = self.easy_data_list + self.hard_data_list
        
        random.shuffle(data_list)
        for data in data_list:
            q_cam = data[0]
            g_cam = data[1]
            q_id = data[2]
            gallery_ids = data[3]
            label = data[4]
            query_track = self.feature_dict[q_cam][q_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))
            tracklets = [query]
            gallery_tracks = list()
            for g_id in gallery_ids:
                gallery_track = torch.tensor(self.feature_dict[g_cam][g_id])
                mean = gallery_track.mean(dim=0)
                std  = gallery_track.std(dim=0, unbiased=False)
                gallery = torch.cat((mean, std))
                gallery_tracks.append(gallery)

            label = torch.tensor([label]).long()
            tracklets.extend(gallery_tracks)
            tracklets_ft = torch.stack(tracklets)
            yield tracklets_ft, label

if __name__ == '__main__':
    easy_file = "/home/apie/projects/MTMC2021/dataset/train/mtmc_train_easy.txt"
    hard_file = "/home/apie/projects/MTMC2021/dataset/train/mtmc_train_hard.txt"
    feature_file = "/home/apie/projects/MTMC2021/dataset/train/gt_features.txt"
    dataset = Dataset(feature_file, easy_file, hard_file)
    dataset_iter = dataset.prepare_data("easy")
    for _ in dataset_iter:
        pass