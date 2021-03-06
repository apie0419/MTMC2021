import os, cv2, random, ffmpeg
import numpy as np
import multiprocessing as mp

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
write_lock  = mp.Lock()

manager = mp.Manager()
colors = manager.dict()
select_cameras = ["c024", "c006", "c027", "c035"]
_type = "sct"

def read_result_file(filename):
    res = dict()
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.split(",")
            if _type == "sct":
                data = list(map(int, data[:6]))
                obj_id = data[1]
                frameid = data[0]
                r = [obj_id]
                r.extend(data[2:])
            elif _type == "mct":
                data = list(map(int, data[:7]))
                obj_id = data[1]
                frameid = data[2]
                r = [obj_id]
                r.extend(data[3:])
            
            if frameid not in res:
                res[frameid] = [r]
            else:
                res[frameid].append(r)
    return res

def compress_video(input_file, output_file):
    
    stream = ffmpeg.input(input_file).video
    stream = ffmpeg.output(stream, output_file)
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, quiet=True)

def prepare_data():
    result_list = list()
    camera_dirs = list()
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    if camera_dir not in select_cameras:
                        continue
                    if _type == "sct":
                        filename = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"{cfg.SCT}_{cfg.DETECTION}_all_features_post.txt")
                    elif _type == "mct":
                        filename = os.path.join(INPUT_DIR, scene_dir, camera_dir, "res_opt.txt")
                    result_dict = read_result_file(filename)
                    result_list.append(result_dict)
                    camera_dirs.append(os.path.join(INPUT_DIR, scene_dir, camera_dir))

    return result_list, camera_dirs

def write_frame(data_list):
    
    vdo_path, results, frame_id = data_list
    cap = cv2.VideoCapture(vdo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if results is not None:
        for r in results:
            id, x, y, w, h = r
            write_lock.acquire()
            color = colors.get(id)
            if color is None:
                color = (random.randint(1, 200), random.randint(1, 200), random.randint(1, 200))
                colors[id] = color
            write_lock.release()
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, str(id), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 4, color, 3)

    return frame

def main(camera_dir, result_dict):
    pool = mp.Pool(NUM_WORKERS)
    
    scene_fd, camera_fd = camera_dir.split('/')[-2:]
    vdo_path = os.path.join(camera_dir, "vdo.avi")
    output_file = os.path.join(camera_dir, f"{camera_fd}_output.avi")
    compress_output_file = os.path.join(camera_dir, f"{camera_fd}_output_compress.avi")

    data = list()
    cap = cv2.VideoCapture(vdo_path)
    framenum = int(cap.get(7))
    resolution = (int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
    for frame_id in range(framenum):
        results = result_dict.get(frame_id)
        data.append([vdo_path, results, frame_id])
        
    for frame in tqdm(pool.imap(write_frame, data), total=len(data), desc=f"Generating Output Video of {scene_fd}/{camera_fd}"):
        out.write(frame)
    
    out.release()
    compress_video(output_file, compress_output_file)
    pool.close()
    pool.join()

if __name__ == "__main__":
    
    print (f"Create {NUM_WORKERS} processes.")
    result_list, camera_dirs = prepare_data()
    
    for i, camera_dir in enumerate(camera_dirs):
        result_dict = result_list[i]
        main(camera_dir, result_dict)
