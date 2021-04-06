import os, torch, time, random
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

device = torch.device(DEVICE + ':' + str(GPU))
model = build_model(cfg, device)
tracklets_file = os.path.join(cfg.PATH.TRAIN_PATH, "gt_features.txt")

_type = "merge"
easy_file = "mtmc_easy_binary.txt"
hard_file = "mtmc_hard_binary.txt"
easy_train_file = os.path.join(cfg.PATH.TRAIN_PATH, easy_file)
hard_train_file = os.path.join(cfg.PATH.TRAIN_PATH, hard_file)
dataset = Dataset(tracklets_file, easy_train_file, hard_train_file, _type)

criterion = build_loss(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model.train()

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

lr = LEARNING_RATE
epochs = EPOCHS

for epoch in range(1, epochs + 1):
    if epoch % 10 == 0:
        lr *= 0.8
        optimizer = Adam(model.parameters(), lr=lr)

    total_map = 0.
    count = 0.
    loss = 0.
    triplet_loss = 0.
    cross_loss = 0.
    triplet_loss_list, cross_loss_list, map_list = list(), list(), list()
    iterations = 1
    if len(dataset) % BATCH_SIZE == 0:
        total = int(len(dataset) / BATCH_SIZE)
    else:
        total = int(len(dataset) / BATCH_SIZE) + 1
    pbar = tqdm(total=total)
    pbar.set_description(f"Epoch {epoch}, Triplet=0, Cross=0, Acc=0%")
    
    for data, target in dataset.prepare_data():
        
        data, target = data.to(device), target.to(device)
        preds, f_prime, fij = model(data)
        triplet, cross = criterion(f_prime, fij, target, preds)
        triplet_loss += triplet.cpu().item()
        cross_loss += cross.cpu().item()
        
        loss += triplet + cross
        
        if (iterations % BATCH_SIZE == 0) or (iterations == len(dataset)):
            loss /= BATCH_SIZE
            triplet_loss /= BATCH_SIZE
            cross_loss /= BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cross_loss_list.append(cross_loss)
            triplet_loss_list.append(triplet_loss)
            _map = total_map / BATCH_SIZE
            map_list.append(_map)
            
            pbar.set_description("Epoch {}, Triplet={:.4f}, Cross={:.4f}, Map={:.2f}%".format(epoch, triplet_loss, cross_loss, _map))
            pbar.update()
            
            loss = 0.
            triplet_loss = 0.
            cross_loss = 0.
            total_map = 0.

        sort_preds = torch.argsort(preds, descending=True)
        target_list = target.cpu().numpy().tolist()
        # print(preds, target)
        for i in range(sort_preds.size(0)):
            if len(target_list) == 0:
                break
            t = sort_preds[i]
            if t in target_list:
                target_list.remove(t)
                point = float((target.size(0) - len(target_list))) / float(i + 1)
                count += point
                
        total_map += (count / target.size(0) * 100.)
        iterations += 1
        count = 0.
        
    avg_map = np.array(map_list).mean()
    avg_cross = np.array(cross_loss_list).mean()
    avg_triplet = np.array(triplet_loss_list).mean()
    pbar.set_description("Epoch {}, Avg_Triplet={:.4f}, Avg_Cross={:.4f}, Avg_Map={:.2f}%".format(epoch, avg_triplet, avg_cross, avg_map))
    pbar.close()
    # if epoch % 10 == 0:
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"mct_epoch{epoch}_{_type}.pth"))
