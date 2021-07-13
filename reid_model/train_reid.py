import sys
sys.path.insert(0,"../")

from options import opt
from datasets.dataset import make_reid_dataset
from pathlib import Path
import torch
from modeling import make_model
import numpy as np
import torch.nn as nn
from losses import make_loss
from tqdm import tqdm
from torch.autograd import Variable
from utils.train_reid_utils import accuracy, calc_mAP
import os
import copy
from solver import make_optimizer, WarmupMultiStepLR
from visual import visualization
from utils.metrics import R1_mAP_eval


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train():
    trainloader, queryloader, galleryloader, num_classes = make_reid_dataset(opt.reid_data_root)
    print(f'image height: {opt.image_height}\t image width: {opt.image_width}')

    model = make_model(num_classes)
    model = model.cuda(opt.cuda_devices)

    loss_func, center_criterion = make_loss(num_classes)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-04, betas=(0.9, 0.999), amsgrad=True)
    optimizer, optimizer_center = make_optimizer(model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, opt.milestones, opt.gamma,
                                  opt.warmup_factor,
                                  opt.warmup_epochs, opt.warmup_method)

    best_mAP = 0.0

    evaluator = R1_mAP_eval(num_classes, max_rank=50, feat_norm=opt.feat_norm,reranking=opt.reranking)

    loss_list = []
    acc_list = []
    mAP_list = []

    for epoch in range(opt.num_epochs):
        print(f'Epoch: {epoch+1}/{opt.num_epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.num_epochs}'))

        total_xentropy_loss = 0.0
        total_triplet_loss = 0.0
        total_center_loss = 0.0
        training_loss = 0.0
        training_corrects = 0
        train_set_size = 0
        
        evaluator.reset()

        model = model.train()
        
        for idx, (images, pids, cam_ids, frameids) in enumerate(tqdm(trainloader)):
            images = images.cuda(opt.cuda_devices)
            pids = pids.cuda(opt.cuda_devices)

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            outputs, features = model(images, pids)

            
            if opt.dataloader_sampler == 'softmax_triplet':
                xentropy_loss, triplet_loss, loss = loss_func(outputs, features, pids)
            elif opt.dataloader_sampler == 'softmax_triplet_center':
                xentropy_loss, triplet_loss, center_loss, loss = loss_func(outputs, features, pids)

            loss.backward()
            optimizer.step()

            if 'center' in opt.metric_loss_type:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / opt.center_loss_weight)
                optimizer_center.step()

            total_xentropy_loss += xentropy_loss.item() * images.size(0)
            total_triplet_loss += triplet_loss.item() * images.size(0)
            if 'center' in opt.metric_loss_type:
                total_center_loss += center_loss.item() * images.size(0)
            training_loss += loss.item() * images.size(0)
            training_corrects += accuracy(outputs, pids)[0] * images.size(0) / 100.0
            train_set_size += images.size(0)
        
        avg_xentropy_loss = total_xentropy_loss / train_set_size
        avg_triplet_loss = total_triplet_loss / train_set_size
        if 'center' in opt.metric_loss_type:
            avg_center_loss =  total_center_loss / train_set_size
        avg_training_loss = training_loss / train_set_size
        avg_training_acc = float(training_corrects) / train_set_size
        if 'center' in opt.metric_loss_type:
            print(f'xentropy_loss: {avg_xentropy_loss:.4f}\ttriplet_loss: {avg_triplet_loss:.4f}\tcenter_loss: {avg_center_loss:.4f}')
        else:
            print(f'xentropy_loss: {avg_xentropy_loss:.4f}\ttriplet_loss: {avg_triplet_loss:.4f}')
        print(f'training_loss: {avg_training_loss:.4f}\ttrain_accuracy: {avg_training_acc:.4f}')

        model.eval()
        # mAP = calc_mAP(queryloader,galleryloader,model)
        for idx, (images, pids, cam_ids,frameids) in enumerate(tqdm(queryloader)):
            with torch.no_grad():
                images = images.cuda(opt.cuda_devices)
                feature = model(images)
                evaluator.update((feature, pids, cam_ids))
        
        for idx, (images, pids, cam_ids,frameids) in enumerate(tqdm(galleryloader)):
            with torch.no_grad():
                images = images.cuda(opt.cuda_devices)
                feature = model(images)
                evaluator.update((feature, pids, cam_ids))

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        print(f'mAP: {mAP}\n')

        scheduler.step()

        loss_list.append(avg_training_loss)
        acc_list.append(avg_training_acc)
        mAP_list.append(mAP)

        if mAP > best_mAP:
            best_mAP = mAP
            best_xentropy_loss = avg_xentropy_loss
            best_triplet_loss = avg_triplet_loss
            if 'center' in opt.metric_loss_type:
                best_center_loss = avg_center_loss
            best_training_loss = avg_training_loss
            best_training_acc = avg_training_acc
            best_model_params = copy.deepcopy(model.state_dict())

        if (epoch+1)%20 == 0:
            model.load_state_dict(best_model_params)
            weight_path = Path(opt.checkpoint_dir).joinpath(f'model-{epoch+1}epoch-{best_mAP:.03f}-mAP.pth')
            torch.save(model,str(weight_path))
            torch.save({'state_dict': model.state_dict()}, str(weight_path)+'.tar')
            # visualization(loss_list, acc_list, mAP_list, epoch+1)
    
    record =  open("record.txt",'w')
    if 'center' in opt.metric_loss_type:
        print(f'best_xentropy_loss: {best_xentropy_loss:.4f}\tbest_triplet_loss: {best_triplet_loss:.4f}\tbest_center_loss: {best_center_loss:.4f}')
        record.write(f'best_xentropy_loss: {best_xentropy_loss:.4f}\tbest_triplet_loss: {best_triplet_loss:.4f}\tbest_center_loss: {best_center_loss:.4f}\n')
    else:
        print(f'best_xentropy_loss: {best_xentropy_loss:.4f}\tbest_triplet_loss: {best_triplet_loss:.4f}')
        record.write(f'best_xentropy_loss: {best_xentropy_loss:.4f}\tbest_triplet_loss: {best_triplet_loss:.4f}\n')
    print(f'best_training_loss: {best_training_loss:.4f}\tbest_accuracy: {best_training_acc:.4f}')
    record.write(f'best_training_loss: {best_training_loss:.4f}\tbest_accuracy: {best_training_acc:.4f}\n')
    print(f'best_mAP: {best_mAP}')
    record.write(f'best_mAP: {best_mAP}')
    record.close()

    model.load_state_dict(best_model_params)
    weight_path = Path(opt.checkpoint_dir).joinpath(f'model-best.pth')
    torch.save(model, str(weight_path))
    torch.save({'state_dict': model.state_dict()}, str(weight_path)+'.tar')
    visualization(loss_list, acc_list, mAP_list, epoch+1)

 
if __name__ == '__main__':
    train()
