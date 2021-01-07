import torch, sys, os, re, time
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset
from PIL              import Image
from tqdm             import tqdm 
from utils.reid       import build_transform, build_model
from utils            import init_path

init_path()

from config import cfg

INPUT_DIR     = cfg.PATH.INPUT_PATH
DEVICE        = cfg.DEVICE.TYPE

if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)

IMS_PER_BATCH = 64

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
        results[l[0]] = ",".join(l[1:])
    
    return results

def _process_data():
    data_queue = mp.Queue()
    max_task_num = 0
    for scene_dir in tqdm(os.listdir(INPUT_DIR), desc="Loading Data"):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"all_features.txt")
            if os.path.exists(feature_file):
                os.remove(feature_file)

            data_dir = os.path.join(INPUT_DIR, scene_dir, camera_dir, "cropped_images")
            img_list = os.listdir(data_dir)
            imgs = [os.path.join(data_dir, img) for img in img_list]
            max_task_num += int(len(imgs)/IMS_PER_BATCH) + 1
            det_feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, "det_feature.txt")
            det_features = read_det_features_file(det_feature_file)
            data_queue.put([imgs, det_features])
            

    return data_queue, max_task_num

def _inference(model, data_queue, finish):
    while not data_queue.empty():
        imgs, det_features = data_queue.get()
        transforms = build_transform(cfg)
        dataset = ImageDataset(imgs, transforms)
        dataloader = DataLoader(dataset, batch_size=IMS_PER_BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)
        with torch.no_grad():
            for data, paths in dataloader:
                data = data.cuda()
                feat = model(data)
                for i,p in enumerate(paths):
                    scene_dir = re.search(r"S([0-9]){2}", p).group(0)
                    camera_dir = re.search(r"c([0-9]){3}", p).group(0)
                    path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
                    
                    with open(os.path.join(path, f'all_features.txt'), 'a+') as f:
                        img_name = p.split('/')[-1]
                        det_feat = det_features[img_name]
                        reid_feat = list(feat[i].cpu().numpy())
                        reid_feat_str = str(reid_feat)[1:-1].replace(" ", "")
                        line = img_name + "," + det_feat + "," + reid_feat_str + "\n"
                        f.write(line)
                finish.value += 1

if __name__ == "__main__":
    mp.set_start_method("spawn")
    model = build_model(cfg)
    model = model.to(DEVICE)
    model = model.eval()
    model.share_memory()
    data_queue, max_task_num = _process_data()
    finish = mp.Value('i', 0)
    
    print (f"Create {cfg.NUM_WORKERS} processes.")
    for i in range(cfg.NUM_WORKERS):
        p = mp.Process(target=_inference, args=(model, data_queue, finish))
        p.start()
    
    for i in tqdm(range(max_task_num), desc="Extracting Features"):
        while i == finish.value:
            time.sleep(0.5)

    data_queue.close()
    data_queue.join_thread()