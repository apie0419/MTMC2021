import argparse
from pathlib import Path

ROOTPATH="/home/apie/projects/MTMC2021_ver2/reid_model"

parser =  argparse.ArgumentParser()

# GENERAL SETTING
parser.add_argument('--cuda-devices', type=int, default=1, help='gpu devices')
parser.add_argument('--checkpoint-dir', type=str, default=str(Path(ROOTPATH).joinpath('checkpoints')), help='directory to save checkpoints')
parser.add_argument('--raw-data-path', type=str, default=str(Path(ROOTPATH).joinpath('../','dataset')), help='Path of original AICITY data')
parser.add_argument('--data-root', type=str, default=str(Path(ROOTPATH).joinpath('Data')), help='Data root path')
parser.add_argument('--test-data', type=str, default=Path(ROOTPATH).joinpath('Data', 'test_data'), help='Root directory of test data')

# REID DATASET
parser.add_argument('--reid-data-root', type=str, default=str(Path(ROOTPATH).joinpath('Data/reid_data')), help='root dir to reid data')
parser.add_argument('--reid-train-batch-size', type=int, default=16, help='training batch size of reid')
parser.add_argument('--reid-train-num-workers', type=int, default=8, help='number of threads for training reid')
parser.add_argument('--reid-test-batch-size', type=int, default=16, help='testing batch size of reid')
parser.add_argument('--reid-test-num-workers', type=int, default=1, help='number of threads for testing reid')
parser.add_argument('--image-height', type=int, default=224, help='height of resizing image')
parser.add_argument('--image-width', type=int, default=224, help='width of resizing image')
parser.add_argument('--test-batch-size', type=int, default=50, help='batch size for testing data')
parser.add_argument('--test-num-workers', type=int, default=4, help='num of workers for testing data')

# REID MODEL
parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--reid-model', type=str, default='resnet101_ibn_a', help='name of reid backbone')
parser.add_argument('--last-stride', type=int, default=1, help='last stride for reid model')
parser.add_argument('--neck', type=str, default='bnneck', help='if reid model train with BNNeck')
parser.add_argument('--test-neck-feat', type=str, default='after', help='Which feature of BNNeck to be used for test, before or after BNNneck, options: [before, after]')
parser.add_argument('--pretrain-choice', type=str, default='', help='Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model, options: [imagenet, self]')
parser.add_argument('--pretrained-dir', type=str, default=str(Path(ROOTPATH).joinpath('checkpoints/pretrained_model')), help='folder which save all the pretrained model')
parser.add_argument('--cos-layer', type=bool, default=False, help='If train with arcface loss, options:[True, False]')
parser.add_argument('--frozen', type=int, default=-1, help='Frozen layers of backbone')

# REID CRITERION
parser.add_argument('--center-loss-weight', type=float, default=0.0005, help='weight for center loss')
parser.add_argument('--metric-loss-type', default='triplet', help='The loss type of metric loss')
parser.add_argument('--no-margin', type=bool, default=False, help='If train with soft triplet loss, options:[True, False]')
parser.add_argument('--margin', type=float, default=0.3, help='Margin of triplet loss')
parser.add_argument('--if-label-smooth', type=str, default='on', help='If train with label smooth, options:[on, off]')
parser.add_argument('--dataloader-sampler', type=str, default='softmax_triplet', help='loss function types')

# OPTIMIZER
parser.add_argument('--base-lr', type=float, default=1e-2, help='base learning rate for reid training')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Settings of weight decay')
parser.add_argument('--center-lr', type=float, default=0.5, help='SGD optimizer learning rate for center loss parameters')
parser.add_argument('--optim-name', type=str, default='SGD', help='optimizer for training re-id model')
parser.add_argument('--bias-lr-factor', default=1, help='Factor of learning bias')
parser.add_argument('--weight_decay_bias', type=float, default=0.0005, help='Settings of weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')

# SCHEDULER
parser.add_argument('--sched-name', type=str, default='MultiStepLR', help='scheduler using for trainig model')
parser.add_argument('--milestones', default=(40,70), help='milestone for re-id scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--warmup-factor', type=float, default=0.01, help='warm up factor')
parser.add_argument('--warmup-epochs', type=int, default=10, help='warm up epochs')
parser.add_argument('--warmup-method', type=str, default='linear', help='method of warm up, option:[constant, linear]')

# EVALUATION
parser.add_argument('--feat_norm', type=str, default='yes', help='Whether feature is nomalized before test, if yes, it is equivalent to cosine distance')
parser.add_argument('--reranking', type=bool, default=False, help='Whether to do reranking')

opt = parser.parse_args()
