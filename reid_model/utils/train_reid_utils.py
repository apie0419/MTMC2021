import torch
import numpy as np
from tqdm import tqdm
from options import opt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        return res


def calc_mAP(queryloader, galleryloader, model):
    model.eval()
    
    query_features = []
    query_pids = []
    query_cam_ids = []

    for idx, (images, pids, cam_ids) in enumerate(tqdm(queryloader)):
        images = images.cuda(opt.cuda_devices)
        feature = model(images)
        feature = feature.data.cpu()
        query_features.append(feature)
        query_pids.extend(pids)
        query_cam_ids.extend(cam_ids)
    
    query_features = torch.cat(query_features, 0)
    query_pids = np.asarray(query_pids)
    query_cam_ids = np.asarray(query_cam_ids)

    gallery_features = []
    gallery_pids = []
    gallery_cam_ids = []

    for idx, (images, pids, cam_ids) in enumerate(tqdm(galleryloader)):
        images = images.cuda(opt.cuda_devices)
        feature = model(images)
        feature = feature.data.cpu()
        gallery_features.append(feature)
        gallery_pids.extend(pids)
        gallery_cam_ids.extend(cam_ids)
    
    gallery_features = torch.cat(gallery_features, 0)
    gallery_pids = np.asarray(gallery_pids)
    gallery_cam_ids = np.asarray(gallery_cam_ids)

    m = query_features.size(0)
    n = gallery_features.size(0)
    
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, query_features, gallery_features.t())
    distmat = distmat.numpy()

    max_rank = 50
    if n < max_rank:
        max_rank = n
    
    indices = np.argsort(distmat, axis=1)

    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(m):
        # get query pid and camid
        q_pid = query_pids[q_idx]
        q_camid = query_cam_ids[q_idx]

        # remove gallery samples that have the same pid and camid with query (basically they are the same)
        order = indices[q_idx]
        remove = (gallery_pids[order] == q_pid) & (gallery_cam_ids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank]) # store all cmc from 0 to 50
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return mAP