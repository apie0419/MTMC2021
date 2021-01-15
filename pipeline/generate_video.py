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

def read_result_file(filename):
    res = dict()
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.split(",")
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

def read_frame(in_file, frame_num):
    out, err = (
        ffmpeg.input(in_file)
              .filter('select', 'gte(n,{})'.format(frame_num))
              .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
              .run(quiet=True)
    )
    image_array = np.asarray(bytearray(out), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def compress_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framenum = int(cap.get(7))
    target_bitrate = int(os.path.getsize(input_file) / 3 / (framenum / fps))
    
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
                    result_dict = read_result_file(os.path.join(INPUT_DIR, scene_dir, camera_dir, "res.txt"))
                    result_list.append(result_dict)
                    camera_dirs.append(os.path.join(INPUT_DIR, scene_dir, camera_dir))

    return result_list, camera_dirs

def write_frame(data_list):
    
    vdo_path, results, frame_id = data_list
    frame = read_frame(vdo_path, frame_id)
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
    compress_vdo_path = os.path.join(camera_dir, "vdo.avi")
    output_file = os.path.join(camera_dir, f"{camera_fd}_output.avi")
    
    data = list()
    cap = cv2.VideoCapture(compress_vdo_path)
    framenum = int(cap.get(7))
    resolution = (int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
    for frame_id in range(framenum):
        results = result_dict.get(frame_id)
        data.append([compress_vdo_path, results, frame_id])
        
    for frame in tqdm(pool.imap(write_frame, data), total=len(data), desc=f"Generating Output Video of {scene_fd}/{camera_fd}"):
        out.write(frame)
    
    out.release()
    pool.close()
    pool.join()

if __name__ == "__main__":
    
    print (f"Create {NUM_WORKERS} processes.")
    result_list, camera_dirs = prepare_data()

    # compress = None
    # while compress==None:
    #     ans = (input("Compress Video?(y/n)") or True)
    #     if ans == 'Y' or ans == 'y':
    #         compress = True
    #     elif ans == 'n' or ans == 'N':
    #         compress = False
    #     else:
    #         print ("Only Y/y or N/n")

    # if compress:
    #     for camera_dir in tqdm(camera_dirs, desc="Compressing Videos"):
    #         vdo_path = os.path.join(camera_dir, "vdo.avi")
    #         compress_vdo_path = os.path.join(camera_dir, "compress_vdo.avi")
    #         compress_video(vdo_path, compress_vdo_path)
    
    for i, camera_dir in enumerate(camera_dirs):
        result_dict = result_list[i]
        main(camera_dir, result_dict)
