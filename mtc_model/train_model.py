import os, torch

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
    num_loss = 0
    pbar = tqdm(dataset.prepare_data(), total=len(dataset))
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        preds, f_prime, fij = model(data)
        loss = criterion(f_prime, fij, target, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_loss = round(loss.item(), 4)
        pbar.set_description(f"Epoch {epoch}, Loss={num_loss}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, f"mtc_epoch{epoch}"))
