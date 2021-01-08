import os, cv2, time
import multiprocessing as mp

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_PATH  = cfg.PATH.INPUT_PATH
IMAGE_SIZE  = cfg.REID.IMG_SIZE
NUM_WORKERS = mp.cpu_count()

class Box(object):
    def __init__(self, id, box):
        self.id = id
        self.box = box

def analysis_to_frame_dict(file_path):
    frame_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        if box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0:
            continue
        cur_box = Box(id, box)
        if index not in frame_dict:
            frame_dict[index] = []
        frame_dict[index].append(cur_box)
    return frame_dict

def prepare_data():
    data_dict = dict()
    scene_fds = os.listdir(INPUT_PATH)
    for scene_fd in scene_fds:

        if scene_fd.startswith("S0"):
            scene_dir = os.path.join(INPUT_PATH, scene_fd)
            camera_fds = os.listdir(scene_dir)

            for camera_fd in camera_fds:

                if camera_fd.startswith('c0'):
                    camera_dir = os.path.join(INPUT_PATH, scene_fd, camera_fd)
                    sct_path = os.path.join(camera_dir, f"mtsc/mtsc_{cfg.SCT}_{cfg.DETECTION}.txt")
                    frame_dict = analysis_to_frame_dict(sct_path)
                    key = scene_fd + '_' + camera_fd
                    if key not in data_dict:
                        data_dict[key] = list()
                    
                    for frame_id in frame_dict:
                        src_boxes = frame_dict[frame_id]
                        data_dict[key].append([frame_id, src_boxes, camera_dir])
                        
    return data_dict

def main(data_list):
    frame_id, src_boxes, camera_dir = data_list
    img_path = os.path.join(camera_dir, "cropped_images", f"{cfg.SCT}_{cfg.DETECTION}")
    video_path = os.path.join(camera_dir, "vdo.avi")
    if not os.path.exists(img_path):
        os.makedirs(img_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret:
        max_y, max_x, _ = frame.shape
        for det_box in src_boxes:
            box = det_box.box
            if box[1] + box[3] > max_y or box[0] + box[2] > max_x:
                continue

            cropped_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            
            img_name = str(frame_id) + '_' + str(det_box.id) + '.jpg'
            out_path = os.path.join(img_path, img_name)
            cv2.imwrite(out_path, cropped_img)

if __name__ == '__main__':
    data_dict = prepare_data()
    pool = mp.Pool(NUM_WORKERS)
    print (f"Create {NUM_WORKERS} processes.")

    for key in data_dict.keys():
        scene_fd, camera_fd = key.split("_")
        data = data_dict[key]

        for _ in tqdm(pool.imap_unordered(main, data), total=len(data), desc=f"Cropping Car Images From {scene_fd}/{camera_fd}"):
            pass

    pool.close()
    pool.join()