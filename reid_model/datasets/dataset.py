from options import opt
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from .random_erasing import RandomErasing



class ReIdDataset(Dataset):
    def __init__(self, root_dir, transform=None,is_training=False):
        self.root_dir =  Path(root_dir)
        self.x = []
        self.pid = []
        self.cam_id = []
        self.frameid = []
        self.transform =  transform
        self.training = is_training

        for image_name in os.listdir(str(self.root_dir)):
            self.x.append(image_name)
            splits = image_name[:-4].split('_')
            self.pid.append(splits[0])
            self.cam_id.append(splits[1])
            self.frameid.append(splits[2])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image_path = Path(self.root_dir).joinpath(self.x[index])
        image =  Image.open(image_path).convert('RGB')
        if self.transform:
            image =  self.transform(image)
        self.pid[index] = int(self.pid[index])
        self.cam_id[index] =  int(self.cam_id[index][1:])
        self.frameid[index] = int(self.frameid[index])
        return image, self.pid[index], self.cam_id[index],self.frameid[index]


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.camid = []
        self.pid = []
        self.frame_id = []
        self.transform = transform
    
        for image_name in os.listdir(str(self.root_dir)):
            self.x.append(image_name)
            splits = image_name[:-4].split('_')
            self.camid.append(splits[0])
            self.pid.append(splits[1])
            self.frame_id.append(splits[2])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        image_path = Path(self.root_dir).joinpath(self.x[index])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        self.camid[index] = int(self.camid[index][1:])
        self.pid[index] = int(self.pid[index])
        self.frame_id[index] = int(self.frame_id[index])
        return image, self.camid[index], self.pid[index], self.frame_id[index]

    
def make_reid_dataset(root_dir):
    data_transform_train = transforms.Compose([
        transforms.Resize((opt.image_height,opt.image_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        RandomErasing()
    ])

    data_transform_valid = transforms.Compose([
        transforms.Resize((opt.image_height,opt.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    trainset = ReIdDataset(Path(opt.reid_data_root).joinpath('train_data'),data_transform_train,is_training=True)
    trainloader =  DataLoader(dataset=trainset, batch_size=opt.reid_train_batch_size, shuffle=True, num_workers=opt.reid_train_num_workers)
    queryset = ReIdDataset(Path(opt.reid_data_root).joinpath('query_data'),data_transform_valid)
    queryloader =  DataLoader(dataset=queryset, batch_size=opt.reid_test_batch_size, shuffle=False, num_workers=opt.reid_test_num_workers)
    galleryset = ReIdDataset(Path(opt.reid_data_root).joinpath('gallery_data'),data_transform_valid)
    galleryloader =  DataLoader(dataset=galleryset, batch_size=opt.reid_test_batch_size, shuffle=False, num_workers=opt.reid_test_num_workers)

    print(f'train size: {len(trainset)}')
    print(f'query size: {len(queryset)}')
    print(f'gallery size: {len(galleryset)}')

    return trainloader, queryloader, galleryloader, len(queryset)+1


def make_testloader():
    root_dir = opt.test_data
    print(root_dir)
    
    data_transform = transforms.Compose([
        transforms.Resize((opt.image_height,opt.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    print(opt.test_batch_size)
    queryset = TestDataset(root_dir=Path(root_dir).joinpath('query_data'), transform=data_transform)
    queryloader = DataLoader(dataset=queryset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.test_num_workers)
    galleryset = TestDataset(root_dir=Path(root_dir).joinpath('gallery_data'), transform=data_transform)
    galleryloader = DataLoader(dataset=galleryset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.test_num_workers)

    return queryloader, galleryloader


