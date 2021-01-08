import os, cv2, time
import multiprocessing as mp

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_PATH = cfg.PATH.INPUT_PATH
IMAGE_SIZE = cfg.REID.IMG_SIZE

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
    scene_dirs = []
    scene_fds = os.listdir(INPUT_PATH)
    max_task_num = 0
    data_queue = mp.Queue()
    for scene_fd in scene_fds:
        if scene_fd.startswith("S01"):
            scene_dirs.append(os.path.join(INPUT_PATH, scene_fd))

    for scene_dir in scene_dirs:
        fds = os.listdir(scene_dir)
        for fd in fds:
            if fd.startswith('c0'):
                camera_dir = os.path.join(scene_dir, fd)
                video_path = os.path.join(camera_dir, 'vdo.avi')
                cap = cv2.VideoCapture(video_path)
                max_task_num += int(cap.get(7))
                data_queue.put(camera_dir)

    return data_queue, max_task_num

def main(data_queue, finish):

    while not data_queue.empty():
        camera_dir = data_queue.get()
        video_path = os.path.join(camera_dir, "vdo.avi")
        sct_path = os.path.join(camera_dir, f"mtsc/mtsc_{cfg.SCT}_{cfg.DETECTION}.txt")
        img_path = os.path.join(camera_dir, "cropped_images", f"{cfg.SCT}_{cfg.DETECTION}")
        if not os.path.exists(img_path):
            os.makedirs(img_path, exist_ok=True)

        frame_dict = analysis_to_frame_dict(sct_path)
        cap = cv2.VideoCapture(video_path)
        all_frames = int(cap.get(7))
        for i in range(all_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            max_y, max_x, _ = frame.shape
            
            if not ret:
                finish.value += 1
                continue

            if i in frame_dict:
                src_boxes = frame_dict[i]

                for det_box in src_boxes:
                    
                    box = det_box.box
                    if box[1] + box[3] > max_y or box[0] + box[2] > max_x:
                        continue

                    cropped_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                    
                    img_name = str(i) + '_' + str(det_box.id) + '.jpg'
                    out_path = os.path.join(img_path, img_name)
                    cv2.imwrite(out_path, cropped_img)
                    
                    
            finish.value += 1


if __name__ == '__main__':
    data_queue, max_task_num = prepare_data()
    finish = mp.Value('i', 0)
    print (f"Create {cfg.NUM_WORKERS} processes.")
    for i in range(cfg.NUM_WORKERS):
        p = mp.Process(target=main, args=(data_queue, finish))
        p.start()
    
    for i in tqdm(range(max_task_num), desc="Cropping Car Images"):
        while i == finish.value:
            time.sleep(0.5)

    data_queue.close()
    data_queue.join_thread()