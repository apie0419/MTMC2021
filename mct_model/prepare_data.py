import os, torch, random

from tqdm        import tqdm
from utils       import init_path
from config      import cfg

init_path()

path = cfg.PATH.VALID_PATH
tracklets_file = os.path.join(path, "gt_features.txt")
easy_output_file = os.path.join(path, "mtmc_easy_binary.txt")
hard_output_file = os.path.join(path, "mtmc_hard_binary.txt")
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
                    g_cameras = list()
                    labels = list()
                    gallery_ids_list = list()
                    count = 0
                    for d in data:
                        label = str(d["label"] + count)
                        g_camera = d["camera"]
                        gallery_ids = d["gallery"]
                        labels.append(label)
                        g_cameras.append(g_camera)
                        count += len(gallery_ids)
                        gallery_ids_list.append(",".join(gallery_ids))
                    if len(labels) > 0:
                        line = q_camera + ' ' + ",".join(g_cameras) + ' ' + q_id + ' ' + "/".join(gallery_ids_list) + ' ' + ','.join(labels) + '\n'
                        f.write(line)
    
def main():
    data_dict = read_feature_file(tracklets_file)
    hard_res_dict = dict()
    easy_res_dict = dict()
    
    for camera_id in tqdm(data_dict, desc="Preparing Data"):
        hard_res_dict[camera_id] = dict()
        easy_res_dict[camera_id] = dict()
        for det_id in data_dict[camera_id]:
            g_cameras = list()
            for camid in data_dict:
                if camid == camera_id:
                    continue
                if det_id in data_dict[camid]:
                    g_cameras.append(camid)
            if len(g_cameras) < 2:
                continue

            hard_res_dict[camera_id][det_id] = list()
            easy_res_dict[camera_id][det_id] = list()
            query_track = data_dict[camera_id][det_id]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            query = torch.cat((mean, std))

            for camid in g_cameras:
                cam_data = data_dict[camid]
                
                # positive
                gallery_track = torch.tensor(cam_data[det_id])
                mean = gallery_track.mean(dim=0)
                std  = gallery_track.std(dim=0, unbiased=False)
                gallery = torch.cat((mean, std))
                num = float(torch.matmul(query, gallery))
                s = torch.norm(query) * torch.norm(gallery)
                if s == 0:
                    pos_cos = 0.0
                else:
                    pos_cos = num/s

                num_objects = 8

                if len(cam_data) - 1 < num_objects:
                    num_objects = len(cam_data) - 1

                # hard sample
                ids = list(cam_data.keys())
                ids.remove(det_id)
                half_num_objects = int(num_objects/2)
                hard_ids = list()
                hard_gallery_tracks = list()

                for id in ids:
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
                        hard_ids.append(id)

                for hid in hard_ids:
                    ids.remove(hid)
                easy_ids = ids
                
                if len(hard_ids) > 1:
                    for i in range(len(hard_ids) / half_num_objects + 1):
                        data = [det_id]
                        hids = hard_ids[i*4:(i+1)*4]
                        data.extend(hids)
                        left = num_objects - len(hids)
                        for eid in easy_ids:
                            if left == 0:
                                data = [det_id]
                                data.extend(hids)
                                hard_gallery_tracks.append(data)
                                left = num_objects - len(hids)
                            data.append(eid)
                            left -= 1

                ## easy sample
                # easy_gallery_tracks = list()
                # data = [det_id]
                # for id in easy_ids:
                #     if num_objects == 0:
                #         break
                    
                #     data.append(id)
                #     num_objects -= 1
                # easy_gallery_tracks.append(data)

                if len(hard_gallery_tracks) > 1:
                    for i in range(len(hard_gallery_tracks)):
                        hard_sample = hard_gallery_tracks[i]
                        orders = list(range(len(hard_sample)))
                        tmp = list(zip(orders, hard_sample))
                        random.shuffle(tmp)
                        orders, hard_sample = zip(*tmp)
                        label = orders.index(0)
                        data = {
                            "gallery": hard_sample,
                            "label": label,
                            "camera": camid
                        }
                        if i > len(hard_res_dict[camera_id][det_id]) - 1:
                            hard_res_dict[camera_id][det_id].append([data])
                        else:
                            hard_res_dict[camera_id][det_id][i].append(data)

                # if len(easy_gallery_tracks) > 1:
                #     for i in range(len(easy_gallery_tracks)):
                #         easy_sample = easy_gallery_tracks[i]
                #         orders = list(range(len(easy_sample)))
                #         tmp = list(zip(orders, easy_sample))
                #         random.shuffle(tmp)
                #         orders, easy_sample = zip(*tmp)
                #         label = orders.index(0)
                #         data = {
                #             "gallery": easy_sample,
                #             "label": label,
                #             "camera": camid
                #         }
                #         if i > len(easy_res_dict[camera_id][det_id]) - 1:
                #             easy_res_dict[camera_id][det_id].append([data])
                #         else:
                #             easy_res_dict[camera_id][det_id][i].append(data)

    write_results(hard_res_dict, hard_output_file)
    # write_results(easy_res_dict, easy_output_file)

if __name__ == "__main__":
    main()