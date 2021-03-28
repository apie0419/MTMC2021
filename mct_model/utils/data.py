import torch, random

from tqdm import tqdm

class Dataset(object):

    def __init__(self, feature_file, easy_tracklets_file, hard_tracklets_file, _type):
        self.feature_dict = self.read_feature_file(feature_file)
        self.easy_data_list = self.read_tracklets_file(easy_tracklets_file)
        self.hard_data_list = self.read_tracklets_file(hard_tracklets_file)
        if _type == "easy":
            self.data_list = self.easy_data_list
        elif _type == "hard":
            self.data_list = self.hard_data_list
        elif _type == "merge":
            self.data_list = self.easy_data_list[:len(self.hard_data_list)] + self.hard_data_list

        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

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
                g_cams = words[1].split(',')
                q_id = words[2]
                g_ids = words[3].split('/')
                labels = list(map(int, words[4].split(',')))
                data_list.append([q_cam, g_cams, q_id, g_ids, labels])
        
        return data_list

    def prepare_data(self):
        
        
        for data in self.data_list:
            q_cam = data[0]
            g_cams = data[1]
            q_id = data[2]
            gallery_ids_list = data[3]
            labels = data[4]
            query_track = self.feature_dict[q_cam][q_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))
            tracklets = [query]
            gallery_tracks = list()
            for i, g_ids in enumerate(gallery_ids_list):
                g_cam = g_cams[i]
                for gid in g_ids.split(','):
                    gallery_track = torch.tensor(self.feature_dict[g_cam][gid])
                    mean = gallery_track.mean(dim=0)
                    std  = gallery_track.std(dim=0, unbiased=False)
                    gallery = torch.cat((mean, std))
                    gallery_tracks.append(gallery)

            label = torch.tensor(labels).long()
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