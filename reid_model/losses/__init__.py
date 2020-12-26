from options import opt
from pathlib import Path
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(num_classes):    # modified by gu
    sampler = opt.dataloader_sampler
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in opt.metric_loss_type:
        if opt.no_margin:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(opt.margin)  # triplet loss
            print("using triplet loss with margin:{}".format(opt.margin))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(opt.metric_loss_type))

    if opt.if_label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, num classes:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target):
            if opt.metric_loss_type == 'triplet':
                #print('using right sampler and loss')
                if opt.if_label_smooth == 'on':
                    xent_loss = xent(score, target)
                    triplet_loss = triplet(feat, target)[0]
                    loss = xent_loss + triplet_loss
                    return xent_loss, triplet_loss, loss
                else:
                    xent_loss = F.cross_entropy(score, target)
                    triplet_loss = triplet(feat, target)[0]
                    loss = xent_loss + triplet_loss
                    return xent_loss, triplet_loss, loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(opt.metric_loss_type))
    elif sampler == 'softmax_triplet_center':
        def loss_func(score, feat, target):
            if opt.if_label_smooth == 'on':
                xent_loss = xent(score, target)
                triplet_loss = triplet(feat, target)[0]
                center_loss = center_criterion(feat, target)
                loss = xent_loss + triplet_loss + opt.center_loss_weight * center_loss
                return xent_loss, triplet_loss, center_loss, loss
            else:
                xent_loss = F.cross_entropy(score, target)
                triplet_loss = triplet(feat, target)[0]
                center_loss = center_criterion(feat, target)
                loss = xent_loss + triplet_loss + opt.center_loss_weight * center_loss
                return xent_loss, triplet_loss, center_loss, loss

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(sampler))
    return loss_func, center_criterion