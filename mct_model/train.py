import os, torch, time, random, argparse
import numpy as np

from tqdm            import tqdm
from utils           import init_path
from utils.data      import Dataset
from modeling        import build_model
from config          import cfg
from losses          import build_loss
from torch.optim     import Adam, SGD, lr_scheduler
from torch.autograd  import Variable
from sklearn.metrics import average_precision_score

torch.autograd.set_detect_anomaly(True)

init_path()
DEVICE        = cfg.DEVICE.TYPE
GPU           = cfg.DEVICE.GPU
LEARNING_RATE = cfg.MCT.LEARNING_RATE
WEIGHT_DECAY  = cfg.MCT.WEIGHT_DECAY
EPOCHS        = cfg.MCT.EPOCHS
BATCH_SIZE    = cfg.MCT.BATCH_SIZE
OUTPUT_PATH   = cfg.PATH.OUTPUT_PATH
RW            = cfg.MCT.RW

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str)
parser.parse_args()
args = parser.parse_args()
output_file_name = args.output

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
merge_dataset = Dataset(tracklets_file, easy_train_file, hard_train_file, "merge")
easy_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "easy")
hard_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "hard")
criterion = build_loss(device)

optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

lr = LEARNING_RATE
epochs = EPOCHS
model.train()

def validation(model, valid_dataset):
    ap_list = list()
    loss_list = list()
    with torch.no_grad():
        for data, target, cams_target, ranked_target in valid_dataset.prepare_data():
            count = 0.
            data, target, cams_target, ranked_target = data.to(device), target.to(device), cams_target.to(device), ranked_target.to(device)
            A, f_prime, cams = model(data)
            
            if RW and A.size(0) > 2:
                preds, g_preds = model.random_walk(A)
                preds = (preds - preds.mean())
                preds = preds * 10
                preds = torch.sigmoid(preds)    
            else:
                preds, g_preds = A[0][1:], A[1:, 1:]
            n = g_preds.size(0)
            g_preds = g_preds.view(n, n)
            g_preds = g_preds.masked_select(~torch.eye(n, dtype=bool))
            all_preds = torch.cat((preds, g_preds))
            cross, camLoss, rankedLoss = criterion(f_prime, target, all_preds, cams, cams_target, ranked_target)
            copy_preds = Variable(preds.clone(), requires_grad=False)
            copy_preds = copy_preds.cpu().numpy()
            target = target.cpu().numpy()[:preds.size(0)]
            ap = average_precision_score(target, copy_preds)
            ap_list.append(ap * 100.)
            loss_list.append(cross.cpu().item())

    _map = np.array(ap_list).mean()
    avg_loss = np.array(loss_list).mean()
    return _map, avg_loss

dataset = merge_dataset
for epoch in range(1, epochs + 1):
    total_ap = 0.
    loss = 0.
    cross_loss = 0.
    cam_loss = 0.
    ranked_loss = 0.
    cross_loss_list, cam_loss_list, ranked_loss_list, ap_list = list(), list(), list(), list()
    iterations = 1
    
    if len(dataset) % BATCH_SIZE == 0:
        total = int(len(dataset) / BATCH_SIZE)
    else:
        total = int(len(dataset) / BATCH_SIZE) + 1
    pbar = tqdm(total=total)
    
    for data, target, cams_target, ranked_target in dataset.prepare_data():
        
        data, target, cams_target, ranked_target = data.to(device), target.to(device), cams_target.to(device), ranked_target.to(device)
        A, f_prime, cams = model(data)
        if RW and A.size(0) > 2:
            preds, g_preds = model.random_walk(A)
            preds = (preds - preds.mean())
            preds = preds * 10
            preds = torch.sigmoid(preds)
        else:
            preds, g_preds = A[0][1:], A[1:, 1:]
        n = g_preds.size(0)
        g_preds = g_preds.view(n, n)
        g_preds = g_preds.masked_select(~torch.eye(n, dtype=bool).to(device))
        all_preds = torch.cat((preds, g_preds))
        cross, camLoss, rankedLoss = criterion(f_prime, target, all_preds, cams, cams_target, ranked_target)
        cross_loss += cross.cpu().item()
        cam_loss += camLoss.cpu().item()
        ranked_loss += rankedLoss.cpu().item() * 0.5
        loss += cross + camLoss + rankedLoss * 0.5
        copy_preds = Variable(preds.clone(), requires_grad=False)
        copy_preds = copy_preds.cpu().numpy()
        target = target.cpu().numpy()[:preds.size(0)]
        ap = average_precision_score(target, copy_preds)
        total_ap += ap * 100.
        if (iterations % BATCH_SIZE == 0) or (iterations == len(dataset)):
            loss /= BATCH_SIZE
            cross_loss /= BATCH_SIZE
            cam_loss /= BATCH_SIZE
            ranked_loss /= BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cross_loss_list.append(cross_loss)
            cam_loss_list.append(cam_loss)
            ranked_loss_list.append(ranked_loss)
            _ap = total_ap / BATCH_SIZE
            ap_list.append(_ap)
            
            pbar.set_description("Epoch {}, Cam={:.4f}, Cross={:.4f}, RankedLoss={:.4f}, Ap={:.2f}%".format(epoch, cam_loss, cross_loss, ranked_loss, _ap))
            pbar.update()
            loss = 0.
            cross_loss = 0.
            cam_loss = 0.
            ranked_loss = 0.
            total_ap = 0.

        iterations += 1

    
    _map = np.array(ap_list).mean()
    avg_cam = np.array(cam_loss_list).mean()
    avg_cross = np.array(cross_loss_list).mean()
    avg_rankedloss = np.array(ranked_loss_list).mean()
    easy_valid_map, easy_valid_loss = validation(model, easy_valid_dataset)
    hard_valid_map, hard_valid_loss = validation(model, hard_valid_dataset)
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"{output_file_name}_{epoch}.pth"))
    pbar.close()
    scheduler.step()
    print("Epoch {}, Avg_Cam={:.4f}, Avg_Cross={:.4f}, Avg_RankedLoss={:.4f}, Map={:.2f}%\nEasy_Valid_Map={:.2f}%, Easy_Valid_Loss={:.4f}, Hard_Valid_Map={:.2f}%, Hard_Valid_Loss={:.4f}".format(epoch, avg_cam, avg_cross, avg_rankedloss, _map, easy_valid_map, easy_valid_loss, hard_valid_map, hard_valid_loss))
    