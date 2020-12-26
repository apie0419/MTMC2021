from utils.reranking import re_ranking
import torch
import numpy as np
from pathlib import Path
import cv2
import copy
from PIL import Image

G_PATH="/mnt/hdd1/home/joycenerd/AIC21-Track3/Data/reid_data/gallery_data"
OUTPATH="/mnt/hdd1/home/joycenerd/AIC21-Track3/Data/demo_reid/match_gallery"


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    print(indices)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        match_idx = []
        for i, val in enumerate(orig_cmc):
            if val==1 and i<50:
                print(f'{i}: {indices[0][i]}')
                match_idx.append(indices[0][i])
    
    return match_idx




class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking=reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.frameids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, frameid= output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.frameids.extend(np.asarray(frameid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:1]
        q_pids = np.asarray(self.pids[:1])
        q_camids = np.asarray(self.camids[:1])
        q_frameids=np.asarray(self.frameids[:1])

        # gallery
        gf = feats[1:]
        g_pids = np.asarray(self.pids[1:])
        g_camids = np.asarray(self.camids[1:])
        g_frameids=np.asarray(self.frameids[1:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)
        match_idx=eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        for i in match_idx:
            pid=g_pids[i]
            camid=g_camids[i]
            frameid=g_frameids[i]

            match_name=str(pid).zfill(5)+'_c'+str(camid).zfill(3)+'_'+str(frameid).zfill(4)+'.jpg'
            match_path=Path(G_PATH).joinpath(match_name)
            print(match_path)
            image=Image.open(match_path)
            out_path=Path(OUTPATH).joinpath(match_name)
            image.save(out_path)