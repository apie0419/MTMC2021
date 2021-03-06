import os, torch, random

from tqdm        import tqdm
from utils       import init_path
from config      import cfg

init_path()

path = cfg.PATH.VALID_PATH
tracklets_file = os.path.join(path, "gt_features.txt")
easy_output_file = os.path.join(path, "mtmc_easy.txt")
hard_output_file = os.path.join(path, "mtmc_hard.txt")
min_gallery_num = 5
max_gallery_num = 15

def read_feature_file(filename):
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

def write_results(res_dict, filename):
    with open(filename, "w+") as f:
        for q_camera in res_dict:
            for q_id in res_dict[q_camera]:
                for data in res_dict[q_camera][q_id]:
                    label = data["label"]
                    g_camera = data["camera"]
                    gallery_ids = data["gallery"]
                    line = q_camera + ' ' + g_camera + ' ' + q_id + ' ' + ",".join(gallery_ids) + ' ' + str(label) + '\n'
                    f.write(line)
    
def main():
    data_dict = read_feature_file(tracklets_file)
    hard_res_dict = dict()
    easy_res_dict = dict()
    for camera_id in tqdm(data_dict, desc="Preparing Data"):
        hard_res_dict[camera_id] = dict()
        easy_res_dict[camera_id] = dict()
        for det_id in data_dict[camera_id]:
            hard_res_dict[camera_id][det_id] = list()
            easy_res_dict[camera_id][det_id] = list()
            query_track = data_dict[camera_id][det_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))

            for camid in data_dict:
                hard_gallery_tracks = list()
                easy_gallery_tracks = list()
                if camid == camera_id:
                    continue
                cam_data = data_dict[camid]
                if det_id not in cam_data:
                    continue
                
                # positive
                gallery_track = torch.tensor(cam_data[det_id])
                mean = gallery_track.mean(dim=0)
                std  = gallery_track.std(dim=0, unbiased=False)
                gallery = torch.cat((mean, std))
                hard_gallery_tracks.append(det_id)
                easy_gallery_tracks.append(det_id)
                num = float(torch.matmul(query, gallery))
                s = torch.norm(query) * torch.norm(gallery)
                if s == 0:
                    pos_cos = 0.0
                else:
                    pos_cos = num/s

                _min = min_gallery_num
                _max = max_gallery_num
                
                if len(cam_data) - 1 < _max:
                    _max = len(cam_data) - 1
                    if len(cam_data) - 1 < _min:
                        _min = len(cam_data) - 1

                num_objects = random.randint(_min, _max)
                ids = list(cam_data.keys())
                random.shuffle(ids)
                half_num_objects = int(num_objects/2)

                # hard sample
                # hard negetive
                for id in ids:
                    if (id == det_id) or (num_objects == half_num_objects):
                        continue
                    gallery_track = torch.tensor(cam_data[id])
                    mean = gallery_track.mean(dim=0)
                    std  = gallery_track.std(dim=0, unbiased=False)
                    gallery = torch.cat((mean, std))

                    num = float(torch.matmul(query, gallery))
                    s = torch.norm(query) * torch.norm(gallery)
                    if s == 0:
                        cos = 0.0
                    else:
                        cos = num/s
                    if cos > pos_cos:
                        hard_gallery_tracks.append(id)
                        num_objects -= 1

                if len(hard_gallery_tracks) > 1:
                    # easy negetive
                    for id in ids:
                        if (num_objects == 0) or (id in hard_gallery_tracks):
                            continue
                        
                        hard_gallery_tracks.append(id)
                        num_objects -= 1

                num_objects = random.randint(_min, _max)
                
                ## easy sample

                for id in ids:
                    if (num_objects == 0) or (id in hard_gallery_tracks):
                        continue
                    
                    easy_gallery_tracks.append(id)
                    num_objects -= 1

                if len(hard_gallery_tracks) > 2:
                    orders = list(range(len(hard_gallery_tracks)))
                    tmp = list(zip(orders, hard_gallery_tracks))
                    random.shuffle(tmp)
                    orders, hard_gallery_tracks = zip(*tmp)
                    label = orders.index(0)
                    data = {
                        "gallery": hard_gallery_tracks,
                        "label": label,
                        "camera": camid
                    }
                    hard_res_dict[camera_id][det_id].append(data)

                if len(easy_gallery_tracks) > 2:
                    orders = list(range(len(easy_gallery_tracks)))
                    tmp = list(zip(orders, easy_gallery_tracks))
                    random.shuffle(tmp)
                    orders, easy_gallery_tracks = zip(*tmp)
                    label = orders.index(0)
                    data = {
                        "gallery": easy_gallery_tracks,
                        "label": label,
                        "camera": camid
                    }
                    easy_res_dict[camera_id][det_id].append(data)

    write_results(hard_res_dict, hard_output_file)
    write_results(easy_res_dict, easy_output_file)

if __name__ == "__main__":
    main()