import os, torch, time, random
import numpy as np

from tqdm        import tqdm
from utils       import init_path
from utils.data  import Dataset
from modeling    import build_model
from config      import cfg
from losses      import build_loss
from torch.optim import Adam

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
tracklets_file = os.path.join(cfg.PATH.INPUT_PATH, "gt_features.txt")
dataset = Dataset(tracklets_file, 5, 15)

criterion = build_loss(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model.train()

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for epoch in range(1, EPOCHS+1):
    if epoch % 10 == 0:
        LEARNING_RATE *= 0.8
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    count = 0.
    loss = 0.
    triplet_loss = 0.
    cross_loss = 0.
    triplet_loss_list, cross_loss_list, acc_list = list(), list(), list()
    iterations = 1
    dataset_len = len(dataset)
    pbar = tqdm(total=int(dataset_len / BATCH_SIZE) + 1)
    pbar.set_description(f"Epoch {epoch}, Triplet=0, Cross=0, Acc=0%")
    
    for data, target in dataset.prepare_data():
        if data == None or target == None:
            dataset_len -= 1
            pbar.total = int(dataset_len / BATCH_SIZE) + 1
            pbar.refresh()
            continue

        data, target = data.to(device), target.to(device)
        preds, f_prime, fij = model(data)
        triplet, cross = criterion(f_prime, fij, target, preds)
        triplet_loss += triplet.item()
        cross_loss += cross.item()
        loss += triplet + cross
        
        if (iterations % BATCH_SIZE == 0) or (iterations == dataset_len):
            loss /= BATCH_SIZE
            triplet_loss /= BATCH_SIZE
            cross_loss /= BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = count / BATCH_SIZE * 100.
            cross_loss_list.append(cross_loss)
            triplet_loss_list.append(triplet_loss)
            acc_list.append(acc)
            pbar.set_description("Epoch {}, Triplet={:.4f}, Cross={:.4f}, Acc={:.2f}%".format(epoch, triplet_loss, cross_loss, acc))
            pbar.update()
            count = 0.
            loss = 0.
            triplet_loss = 0.
            cross_loss = 0.
        
        if preds.argmax().item() == target[0].item():
            count += 1
        
        iterations += 1
        
    avg_acc = np.array(acc_list).mean()
    avg_cross = np.array(cross_loss_list).mean()
    avg_triplet = np.array(triplet_loss_list).mean()
    pbar.set_description("Epoch {}, Avg_Triplet={:.4f}, Avg_Cross={:4f}, Avg_Acc={:.2f}%".format(epoch, avg_triplet, avg_cross, avg_acc))
    pbar.close()
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"mct_epoch{epoch}.pth"))