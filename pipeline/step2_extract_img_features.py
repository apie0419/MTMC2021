import torch, os, re, time
import torch.multiprocessing as mp
import numpy as np

from torch.utils.data import DataLoader, Dataset
from PIL              import Image
from tqdm             import tqdm 
from utils.reid       import build_transform, build_model
from utils            import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR     = cfg.PATH.INPUT_PATH
DEVICE        = cfg.DEVICE.TYPE
GPU           = cfg.DEVICE.GPU
NUM_WORKERS   = 4
IMS_PER_BATCH = 64
device        = None

if DEVICE == "cuda":
    device = torch.device(DEVICE + ':' + str(GPU))
elif DEVICE == "cpu":
    device = torch.device(DEVICE)

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def read_image(self, img_path):
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                exit()
        return img

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, img_path

def collate_fn(batch):
    imgs, path = zip(*batch)
    return torch.stack(imgs, dim=0), path

def read_det_features_file(file):
    f = open(file, "r")
    results = dict()
    for line in f.readlines():
        l = line.strip("\n").split(",")
        frame_id = l[0]
        id = l[1]
        box = ",".join(l[2:])
        results[frame_id + '_' + id] = box
    
    return results

def analysis_transfrom_mat(cali_path):
    first_line = open(cali_path).readlines()[0].strip('\r\n')
    cols = first_line.lstrip('Homography matrix: ').split(';')
    transfrom_mat = np.ones((3, 3))
    for i in range(3):
        values_string = cols[i].split()
        for j in range(3):
            value = float(values_string[j])
            transfrom_mat[i][j] = value
    inv_transfrom_mat = np.linalg.inv(transfrom_mat)
    return inv_transfrom_mat

def prepare_data():
    data_queue = mp.Queue()
    max_task_num = 0
    for scene_dir in tqdm(os.listdir(INPUT_DIR), desc="Loading Data"):
        if not scene_dir.startswith("S01"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"all_features.txt")
            if os.path.exists(feature_file):
                os.remove(feature_file)

            cali_path = os.path.join(INPUT_DIR, scene_dir, camera_dir, 'calibration.txt')
            trans_mat = analysis_transfrom_mat(cali_path)
            data_dir = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"cropped_images/{cfg.SCT}_{cfg.DETECTION}")
            img_list = os.listdir(data_dir)
            imgs = [os.path.join(data_dir, img) for img in img_list]
            max_task_num += len(imgs)
            det_feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"mtsc/mtsc_{cfg.SCT}_{cfg.DETECTION}.txt")
            det_features = read_det_features_file(det_feature_file)
            data_queue.put([imgs, det_features, trans_mat])
            

    return data_queue, max_task_num

def main(model, data_queue, finish):
    
    while not data_queue.empty():
        imgs, det_features, trans_mat = data_queue.get()
        transforms = build_transform(cfg)
        dataset = ImageDataset(imgs, transforms)
        dataloader = DataLoader(dataset, batch_size=IMS_PER_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
        with torch.no_grad():
            for data, paths in dataloader:
                
                data = data.to(device)
                feat = model(data)
                for i,p in enumerate(paths):
                    scene_dir = re.search(r"S([0-9]){2}", p).group(0)
                    camera_dir = re.search(r"c([0-9]){3}", p).group(0)
                    path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
                    
                    with open(os.path.join(path, f'all_features.txt'), 'a+') as f:
                        key = p.split('/')[-1][:-4]
                        det_feat = det_features[key]
                        frame_id, id = key.split("_")
                        box = det_feat.split(",")
                        coor = [int(float(box[0])) + int(float(box[2]))/2, int(float(box[1])) + int(float(box[3]))/2, 1]
                        GPS_coor = np.dot(trans_mat, coor)
                        GPS_coor = GPS_coor / GPS_coor[2]
                        reid_feat = list(feat[i].cpu().numpy())
                        reid_feat_str = str(reid_feat)[1:-1].replace(" ", "")
                        line = frame_id + "," + id + "," + det_feat + "," + str(GPS_coor[0]) + "," + str(GPS_coor[1]) \
                                + "," + reid_feat_str + "\n"
                        f.write(line)
                finish.value += len(paths)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    model = build_model(cfg)
    model.to(device)
    model = model.eval()
    model.share_memory()
    data_queue, max_task_num = prepare_data()
    finish = mp.Value('i', 0)
    
    print (f"Create {NUM_WORKERS} processes.")
    for i in range(NUM_WORKERS):
        p = mp.Process(target=main, args=(model, data_queue, finish))
        p.start()
    
    for i in tqdm(range(max_task_num), desc="Extracting Features"):
        while i == finish.value:
            time.sleep(0.1)

    data_queue.close()
    data_queue.join_thread()