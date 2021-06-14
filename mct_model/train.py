import os, torch, time, random, argparse
import numpy as np

from tqdm        import tqdm
from utils       import init_path
from utils.data  import Dataset
from modeling    import build_model
from config      import cfg
from losses      import build_loss
from torch.optim import Adam, SGD

torch.autograd.set_detect_anomaly(True)

init_path()
DEVICE        = cfg.DEVICE.TYPE
GPU           = cfg.DEVICE.GPU
LEARNING_RATE = cfg.MCT.LEARNING_RATE
EPOCHS        = cfg.MCT.EPOCHS
BATCH_SIZE    = cfg.MCT.BATCH_SIZE
OUTPUT_PATH   = cfg.PATH.OUTPUT_PATH
RW            = cfg.MCT.RW

device = torch.device(DEVICE + ':' + str(GPU))
model = build_model(cfg, device)
tracklets_file = os.path.join(cfg.PATH.TRAIN_PATH, "gt_features.txt")
valid_tracklet_file = os.path.join(cfg.PATH.VALID_PATH, "gt_features.txt")
easy_file = "mtmc_easy_binary_multicam.txt"
hard_file = "mtmc_hard_binary_multicam.txt"
easy_train_file = os.path.join(cfg.PATH.TRAIN_PATH, easy_file)
hard_train_file = os.path.join(cfg.PATH.TRAIN_PATH, hard_file)
easy_valid_file = os.path.join(cfg.PATH.VALID_PATH, easy_file)
hard_valid_file = os.path.join(cfg.PATH.VALID_PATH, hard_file)
easy_dataset = Dataset(tracklets_file, easy_train_file, hard_train_file, "easy", training=True)
merge_dataset = Dataset(tracklets_file, easy_train_file, hard_train_file, "merge", training=True)
easy_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "easy", training=False)
hard_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "hard", training=False)
criterion = build_loss(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str)
parser.parse_args()
args = parser.parse_args()

output_file_name = args.output

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

lr = LEARNING_RATE
epochs = EPOCHS
model.train()

def validation(model, valid_dataset, epoch):
    ap_list = list()
    loss_list = list()
    with torch.no_grad():
        for data, target, cams_target in valid_dataset.prepare_data():
            count = 0.
            data, target, cams_target = data.to(device), target.to(device), cams_target.to(device)
            A, f_prime, fij, cams = model(data)
            
            if epoch >= 1 and RW:
                preds = model.random_walk(A)
            else:
                preds = A[0][1:]
            # preds = model.random_walk(A)
            preds = (preds - preds.mean())
            preds = preds * 100
            preds = torch.sigmoid(preds)
            triplet, cross, camLoss = criterion(f_prime, fij, target, preds, cams, cams_target)
            sort_preds = torch.argsort(preds, descending=True)
            target_list = target.cpu().numpy().tolist()
            for i in range(sort_preds.size(0)):
                if len(target_list) == 0:
                    break
                t = sort_preds[i]
                if t in target_list:
                    target_list.remove(t)
                    point = float((target.size(0) - len(target_list))) / float(i + 1)
                    count += point
            ap_list.append(count / target.size(0) * 100.)
            loss_list.append(cross.cpu().item())

    _map = np.array(ap_list).mean()
    avg_loss = np.array(loss_list).mean()
    return _map, avg_loss

for epoch in range(1, epochs + 1):
    if epoch >= 6:
        dataset = merge_dataset
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE * 0.5, momentum=0.9)
    else:
        dataset = merge_dataset
    total_ap = 0.
    count = 0.
    loss = 0.
    triplet_loss = 0.
    cross_loss = 0.
    cam_loss = 0.
    triplet_loss_list, cross_loss_list, cam_loss_list, ap_list = list(), list(), list(), list()
    iterations = 1
    
    if len(dataset) % BATCH_SIZE == 0:
        total = int(len(dataset) / BATCH_SIZE)
    else:
        total = int(len(dataset) / BATCH_SIZE) + 1
    pbar = tqdm(total=total)
    pbar.set_description(f"Epoch {epoch}, Triplet=0, Cross=0, Acc=0%")
    
    for data, target, cams_target in dataset.prepare_data():
        
        data, target, cams_target = data.to(device), target.to(device), cams_target.to(device)
        A, f_prime, fij, cams = model(data)
        
        if epoch >= 1 and RW:
            preds = model.random_walk(A)
        else:
            preds = A[0][1:]
        # preds = model.random_walk(A)
        preds = (preds - preds.mean())
        preds = preds * 100
        preds = torch.sigmoid(preds)
        triplet, cross, camLoss = criterion(f_prime, fij, target, preds, cams, cams_target)
        triplet_loss += triplet.cpu().item()
        cross_loss += cross.cpu().item()
        cam_loss += camLoss.cpu().item()
        loss += cross
        
        if (iterations % BATCH_SIZE == 0) or (iterations == len(dataset)):
            loss /= BATCH_SIZE
            triplet_loss /= BATCH_SIZE
            cross_loss /= BATCH_SIZE
            cam_loss /= BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cross_loss_list.append(cross_loss)
            triplet_loss_list.append(triplet_loss)
            cam_loss_list.append(cam_loss)
            _ap = total_ap / BATCH_SIZE
            ap_list.append(_ap)
            
            pbar.set_description("Epoch {}, Triplet={:.4f}, Cross={:.4f}, Ap={:.2f}%".format(epoch, triplet_loss, cross_loss, _ap))
            pbar.update()
            
            loss = 0.
            triplet_loss = 0.
            cross_loss = 0.
            cam_loss = 0.
            total_ap = 0.

        sort_preds = torch.argsort(preds, descending=True)
        target_list = target.cpu().numpy().tolist()
        for i in range(sort_preds.size(0)):
            if len(target_list) == 0:
                break
            t = sort_preds[i]
            if t in target_list:
                target_list.remove(t)
                point = float((target.size(0) - len(target_list))) / float(i + 1)
                count += point
                
        total_ap += (count / target.size(0) * 100.)
        
        iterations += 1
        count = 0.
        
    _map = np.array(ap_list).mean()
    avg_cam = np.array(cam_loss_list).mean()
    avg_cross = np.array(cross_loss_list).mean()
    avg_triplet = np.array(triplet_loss_list).mean()
    easy_valid_map, easy_valid_loss = validation(model, easy_valid_dataset, epoch)
    hard_valid_map, hard_valid_loss = validation(model, hard_valid_dataset, epoch)
    # easy_valid_map, easy_valid_loss = 0, 0
    # hard_valid_map, hard_valid_loss = 0, 0
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"{output_file_name}_{epoch}.pth"))
    pbar.close()
    print("Epoch {}, Avg_Triplet={:.4f}, Avg_Cam={:.4f}, Avg_Cross={:.4f}, Map={:.2f}%\nEasy_Valid_Map={:.2f}%, Easy_Valid_Loss={:.4f}, Hard_Valid_Map={:.2f}%, Hard_Valid_Loss={:.4f}".format(epoch, avg_triplet, avg_cam, avg_cross, _map, easy_valid_map, easy_valid_loss, hard_valid_map, hard_valid_loss))
    