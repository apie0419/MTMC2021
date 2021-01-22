import os, torch, time

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
GPUS          = cfg.DEVICE.GPUS
LEARNING_RATE = cfg.MCT.LEARNING_RATE
EPOCHS        = cfg.MCT.EPOCHS
BATCH_SIZE    = cfg.MCT.BATCH_SIZE
OUTPUT_PATH   = cfg.PATH.OUTPUT_PATH

device = torch.device(DEVICE + ':' + str(GPUS[0]))
model = build_model(cfg, device)
tracklets_file = os.path.join(cfg.PATH.INPUT_PATH, "gt_features.txt")
dataset = Dataset(tracklets_file)

criterion = build_loss(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model = model.to(device)
model = model.train()

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for epoch in range(1, EPOCHS+1):
    count = 0.
    loss = 0.
    iterations = 1
    dataset_len = len(dataset)
    pbar = tqdm(total=int(dataset_len / BATCH_SIZE) + 1)
    pbar.set_description(f"Epoch {epoch}, Loss=0, Acc=0%")

    for data, target in dataset.prepare_data():
        if data == None or target == None:
            dataset_len -= 1
            pbar.total = int(dataset_len / BATCH_SIZE) + 1
            pbar.refresh()
            continue

        data, target = data.to(device), target.to(device)
        preds, f_prime, fij = model(data)
        
        loss += criterion(f_prime, fij, target, preds)
        
        if (iterations % BATCH_SIZE == 0) or (iterations == dataset_len):
            loss /= BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = round(count / BATCH_SIZE * 100., 2)
            num_loss = round(loss.item(), 4)
            pbar.set_description(f"Epoch {epoch}, Loss={num_loss}, Acc={acc}%")
            pbar.update()
            count = 0.
            loss = 0.
        
        if preds.argmax().item() == target[0].item():
            count += 1
        
        iterations += 1
        
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"mtc_epoch{epoch}.pth"))
