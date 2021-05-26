import torch, random

from tqdm import tqdm

class Dataset(object):

    def __init__(self, feature_file, easy_tracklets_file, hard_tracklets_file, _type, training=True):
        self.feature_dict = self.read_feature_file(feature_file)
        self.easy_data_list = self.read_tracklets_file(easy_tracklets_file)
        self.hard_data_list = self.read_tracklets_file(hard_tracklets_file)
        self._type = _type
        self.training = training
        if self.training:
            if _type == "easy":
                self.data_list = self.easy_data_list
            elif _type == "hard":
                self.data_list = self.hard_data_list
            elif _type == "merge":
                self.data_list = self.easy_data_list[:len(self.hard_data_list)] + self.hard_data_list
        else:
            if _type == "easy":
                self.data_list = self.easy_data_list
            elif _type == "hard":
                self.data_list = self.hard_data_list
            elif _type == "merge":
                self.data_list = self.easy_data_list[:len(self.hard_data_list) * 9] + self.hard_data_list
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
                g_cam = words[1]
                q_id = words[2]
                g_ids = words[3]
                label = int(words[4])
                data_list.append([q_cam, g_cam, q_id, g_ids, label])
        
        return data_list

    def prepare_data(self):
        if self.training and self._type == "merge":
            random.shuffle(self.easy_data_list)
            self.data_list = self.easy_data_list[:len(self.hard_data_list)] + self.hard_data_list
            random.shuffle(self.data_list)

        for data in self.data_list:
            q_cam = data[0]
            g_cam = data[1]
            q_id = data[2]
            g_ids = data[3]
            label = data[4]
            qcam = int(q_cam[1:])
            qcam -= 1
            if qcam > 4:
                qcam -= 4
            cam_label = [qcam]
            gcam = int(g_cam[1:])
            gcam -= 1
            if gcam > 5:
                gcam -= 4
            
            query_track = self.feature_dict[q_cam][q_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))
            tracklets = [query]
            gallery_tracks = list()
            for gid in g_ids.split(","):
                gallery_track = torch.tensor(self.feature_dict[g_cam][gid])
                mean = gallery_track.mean(dim=0)
                std  = gallery_track.std(dim=0, unbiased=False)
                gallery = torch.cat((mean, std))
                gallery_tracks.append(gallery)
                cam_label.append(gcam)
            label = torch.tensor(label).long()
            cam_label = torch.tensor(cam_label).long()
            tracklets.extend(gallery_tracks)
            tracklets_ft = torch.stack(tracklets)

            yield tracklets_ft, label, cam_label


if __name__ == '__main__':
    easy_file = "/home/apie/projects/MTMC2021/dataset/train/mtmc_train_easy.txt"
    hard_file = "/home/apie/projects/MTMC2021/dataset/train/mtmc_train_hard.txt"
    feature_file = "/home/apie/projects/MTMC2021/dataset/train/gt_features.txt"
    dataset = Dataset(feature_file, easy_file, hard_file)
    dataset_iter = dataset.prepare_data("easy")
    for _ in dataset_iter:
        pass