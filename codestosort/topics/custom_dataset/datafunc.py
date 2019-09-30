import os
import torch
import torchvision

from PIL import Image
from tqdm import tqdm

# example
# 378.0 318.0 401.0 318.0 401.0 386.0 378.0 386.0 large-vehicle 0
# 401.0 289.0 429.0 289.0 429.0 386.0 401.0 386.0 large-vehicle 0
# 435.0 336.0 458.0 336.0 458.0 393.0 435.0 393.0 large-vehicle 0
# 509.0 363.0 512.0 363.0 512.0 401.0 509.0 401.0 small-vehicle 2


# data_dir_root should be hw2_train_val
# train directory: hw2_train_val/train15000 # 00000 to 14999   
# val directory: hw2_train_val/val1500  # 0000 to 1499
#       sub dir: images   labelTxt_hbb

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def YoloDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir_root, train=True, transform=None, target_transform=None, create_file=False):

        self.data_dir_root = data_dir_root
        self.transform = transform
        self.train = train

        self.train_file = os.path.join(data_dir_root, 'train.pt')
        self.valid_file = os.path.join(data_dir_root, 'valid.pt')

        if create_file:
            self.create_data_file()

        if not self._check_exist():
            raise RuntimeError('Dataset not created.' +
                               ' You can use create_file=True to create it')

        if self.train:
            self.train_data, self.train_labels = torch.load(self.train_file)
        else:
            self.valid_data, self.valid_labels = torch.load(self.valid_file)

    def __len__(self):

        if self.train:
            return 15000
        else:
            return 1500

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.valid_data[index], self.valid_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def _check_exist(self):

        return os.path.exists(self.train_file) and os.path.exists(self.valid_file)

    def create_data_file(self): ## load data, save as torch tensor

        train_dir = os.path.join(self.data_dir_root, 'train15000')
        valid_dir = os.path.join(self.data_dir_root, 'valid1500')

        for root_dir, save_path in zip([train_dir, valid_dir],[self.train_file, self.valid_file]):
            
            print('Loading data from', root_dir)

            image_dir = os.path.join(root_dir, 'images')
            label_dir = os.path.join(root_dir, 'labelTxt_hbb')
            total = len(os.listdir(image_dir))
            trans = torchvision.trainsforms.Compose([
                torchvision.transforms.Resize(512),
                torchvision.transforms.ToTensor(),
            ])
            img_set = []
            target_set = []
            for i in tqdm(range(total)):
                k = len(str(total))
                def str_(i):
                    return '0'*(k-len(str(i)))+str(i)

                img_path = os.path.join(image_dir, str_(i)+'.jpg')
                img = Image.open(img_path)
                img = trans(img)
                img_set.append(img)

                h, w, __ = img.shape

                label_path = os.path.join(label_dir, str_(i)+'.txt')
                target = reader(label_path, h, w)
                target_set.append(target)

            img_set = torch.cat(img_set)
            target_set = torch.cat(target_set)
            torch.save((img_set, target_set), save_path)

            print('saved to', save_path)

    @staticmethod
    def reader(label_path, h, w): # read file and output target

        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            obj_ = line.strip().split()
            xmin = float(obj_[0])
            ymin = float(obj_[1])
            xmax = float(obj_[4])
            ymax = float(obj_[5])
            obj_class = classnames.index(obj_[8]) + 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_class)        ### +1???

        boxes = torch.Tensor(boxes)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        labels = torch.LongTensor(labels)

        return encoder(boxes, labels)

    @staticmethod
    def encoder(boxes, labels)

        target = torch.zeros((7,7,26))
        cell_size = 1./7
        width_height = boxes[:,2:] - boxes[:,:2]
        center_x_ys = (boxes[:,2:]+ boxes[:,:2])/2
        for it, center_x_y in enumerate(center_x_ys):
            cell_i_j = (center_x_y/cell_size).ceil() - 1
            i = int(cell_i_j[1])
            j = int(cell_i_j[0])
            target[i, j, 4] = 1
            target[i, j, 9] = 1
            target[i, j, int(labels[it]+9)] = 1
            tl_x_y = cell_i_j * cell_size
            delta_x_y = (center_x_y - tl_x_y) / cell_size
            target[i, j, 2:4] = width_height[it]
            target[i, j, :2] = delta_x_y
            target[i, j, 7:9] = width_height[it]
            target[i, j, 5:7] = delta_x_y

            ## probably no overlapse  ## (center delta x,y; width, heigth, probability) x2, 16 class ... for 1 cell

        return target


## can implement random crop etc later

## (x y x y x y x y class level) -> (7x7x26) written above
## (x y x y x y x y class level) <- (7x7x26) to be implemented in (predict)

def make_dataloader(data_dir_root, img_size=448, batch_size=128):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
        ])

    trainset = YoloDataset(data_dir_root, train=True, transform=trans, create_file=True)
    validset = YoloDataset(data_dir_root, train=False, transform=trans, create_file=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    return trainloader, validloader