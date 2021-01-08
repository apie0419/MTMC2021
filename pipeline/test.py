import torch, os, re, time
import multiprocessing as mp
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
BATCH_SIZE    = cfg.REID.BATCH_SIZE
NUM_WORKERS   = mp.cpu_count()
STOP          = mp.Value('i', False)
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

def writer(data_queue, scene_dir, camera_dir):
    feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"all_features.txt")
    cali_path = os.path.join(INPUT_DIR, scene_dir, camera_dir, 'calibration.txt')
    det_feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"mtsc/mtsc_{cfg.SCT}_{cfg.DETECTION}.txt")
    trans_mat = analysis_transfrom_mat(cali_path)
    det_features = read_det_features_file(det_feature_file)

    with open(feature_file, 'w+') as f:
        while not STOP.value or not data_queue.empty():
            if data_queue.empty():
                continue
            img_path, feat = data_queue.get()
            key = img_path.split('/')[-1][:-4]
            det_feat = det_features[key]
            frame_id, id = key.split("_")
            box = det_feat.split(",")
            coor = [int(float(box[0])) + int(float(box[2]))/2, int(float(box[1])) + int(float(box[3]))/2, 1]
            GPS_coor = np.dot(trans_mat, coor)
            GPS_coor = GPS_coor / GPS_coor[2]
            reid_feat = list(feat)
            reid_feat_str = str(reid_feat)[1:-1].replace(" ", "")
            line = frame_id + "," + id + "," + det_feat + "," + str(GPS_coor[0]) + "," + str(GPS_coor[1]) \
                    + "," + reid_feat_str + "\n"
            f.write(line)

def prepare_data():
    dirs = list()
    imgs = list()
    for scene_dir in tqdm(os.listdir(INPUT_DIR), desc="Loading Data"):
        if not scene_dir.startswith("S01"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue  
            img_dir = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"cropped_images/{cfg.SCT}_{cfg.DETECTION}")
            imgs.extend([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
            dirs.append([scene_dir, camera_dir])
            
            
    return imgs, dirs

def main(model, imgs, queue_dict):
    transforms = build_transform(cfg)
    dataset = ImageDataset(imgs, transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    with torch.no_grad():
        for data, paths in tqdm(dataloader, desc="Extracting Features"):
            data = data.to(device)
            feat = model(data)

            for i, img_path in enumerate(paths):
                scene_dir = re.search(r"S([0-9]){2}", img_path).group(0)
                camera_dir = re.search(r"c([0-9]){3}", img_path).group(0)
                key = scene_dir + '_' + camera_dir
                queue = queue_dict[key]
                queue.put([img_path, feat[i].cpu().numpy()])
                    

if __name__ == "__main__":
    model = build_model(cfg)
    model.to(device)
    model = model.eval()
    imgs, dirs = prepare_data()
    
    print (f"Create {len(dirs)} processes.")
    queue_dict = dict()

    for scene_dir, camera_dir in dirs:
        key = scene_dir + '_' + camera_dir
        queue = mp.Queue()
        queue_dict[key] = queue
        p = mp.Process(target=writer, args=(queue, scene_dir, camera_dir))
        p.start()
            
    main(model, imgs, queue_dict)
    STOP.value = True

    for key in queue_dict:
        queue = queue_dict[key]
        queue.close()
        queue.join_thread()