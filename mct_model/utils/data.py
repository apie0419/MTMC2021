import torch, random

from tqdm import tqdm

class Dataset(object):

    def __init__(self, feature_file, easy_tracklets_file, hard_tracklets_file, _type):
        self.feature_dict = self.read_feature_file(feature_file)
        self._type = _type
        if _type == "easy":
            self.easy_data_list = self.read_tracklets_file(easy_tracklets_file)
            self.data_list = self.easy_data_list
        elif _type == "hard":
            self.hard_data_list = self.read_tracklets_file(hard_tracklets_file)
            self.data_list = self.hard_data_list
        elif _type == "merge":
            self.easy_data_list = self.read_tracklets_file(easy_tracklets_file)
            self.hard_data_list = self.read_tracklets_file(hard_tracklets_file)
            self.data_list = self.easy_data_list + self.hard_data_list
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
                q_id = words[1]
                g_ids = list()
                labels = list()
                for i, word in enumerate(words[2].split(',')):
                    gcam, g_id = word.split('/')
                    if g_id == q_id:
                        labels.append(1)
                    else:
                        labels.append(0)
                    g_ids.append([gcam, g_id])
                data_list.append([q_cam, q_id, g_ids, labels])
        
        return data_list

    def prepare_data(self):
        
        for data in self.data_list:
            q_cam = data[0]
            q_id = data[1]
            g_ids = data[2]
            labels = data[3]
            qcam = int(q_cam[1:])
            qcam -= 1
            if qcam > 4:
                qcam -= 4
            cam_label = [qcam]
            ap_labels = [int(q_id)]
            query_track = self.feature_dict[q_cam][q_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))
            tracklets = [query]
            gallery_tracks = list()
            for gcam, g_id in g_ids:
                ap_labels.append(int(g_id))
                gallery_track = torch.tensor(self.feature_dict[gcam][g_id])
                mean = gallery_track.mean(dim=0)
                std  = gallery_track.std(dim=0, unbiased=False)
                gallery = torch.cat((mean, std))
                gallery_tracks.append(gallery)
                gcam = int(gcam[1:])
                gcam -= 1
                if gcam > 4:
                    gcam -= 4
                cam_label.append(gcam)

            labels = torch.tensor(labels).float()
            tmp_labels = list()
            tmp = ap_labels[1:]
            n = len(tmp)
            for i in range(n):
                label = torch.tensor(tmp)
                label = torch.where(label==label[i], 1, 0)
                tmp_labels.append(label)
            tmp_labels = torch.stack(tmp_labels)
            tmp_labels = tmp_labels.masked_select(~torch.eye(n, dtype=bool))
            labels = torch.cat((labels, tmp_labels)).float()
            ap_labels = torch.tensor(ap_labels).long()
            cam_label = torch.tensor(cam_label).long()
            tracklets.extend(gallery_tracks)
            tracklets_ft = torch.stack(tracklets)

            yield tracklets_ft, labels, cam_label, ap_labels


if __name__ == '__main__':
    easy_file = "/home/apie/projects/MTMC2021_ver2/dataset/train/mtmc_easy_binary_multicam.txt"
    hard_file = "/home/apie/projects/MTMC2021_ver2/dataset/train/mtmc_hard_binary_multicam.txt"
    feature_file = "/home/apie/projects/MTMC2021_ver2/dataset/train/gt_features.txt"
    dataset = Dataset(feature_file, easy_file, hard_file, "easy")
    dataset_iter = dataset.prepare_data()
    for _ in dataset_iter:
        pass