import os
import torch
import torchvision

import cv2
from PIL import Image
from tqdm import tqdm

import gc

import sys
sys.path.append('module')

if __name__ == '__main__':
    from augment import data_augmentation, processimg
    from prediction import encoder, reader, _decoder
else:    
    from module.augment import data_augmentation, processimg
    from module.prediction import encoder, reader, _decoder

# example
# 378.0 318.0 401.0 318.0 401.0 386.0 378.0 386.0 large-vehicle 0
# 401.0 289.0 429.0 289.0 429.0 386.0 401.0 386.0 large-vehicle 0
# 435.0 336.0 458.0 336.0 458.0 393.0 435.0 393.0 large-vehicle 0
# 509.0 363.0 512.0 363.0 512.0 401.0 509.0 401.0 small-vehicle 2


# data_dir_root should be hw2_train_val
# train directory: hw2_train_val/train15000 # 00000 to 14999   
# val directory: hw2_train_val/val1500  # 0000 to 1499
#       sub dir: images   labelTxt_hbb

# RAM will explode if save all tensor to single file; save each tensor to directory seperately!


# classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

classnames = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane']



class YoloDataset(torch.utils.data.Dataset):  ## for train and valid only

    def __init__(self, root_dir, augment=True, transform=None):

        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labelTxt_hbb')

        self.augment = augment        
        self.transform = transform

        self.length = len(os.listdir(self.image_dir))
        k = len(str(self.length))
        self.str_ =  lambda i: '0' * (k - len(str(i))) + str(i)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self._readlabel()

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        img = self._readimg(index)
        boxes = self.box_array[index].clone()
        labels = self.label_array[index].clone()

        if self.augment:
            img, boxes, labels = data_augmentation(img, boxes, labels)

        h, w, __ = img.shape
        # boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes).to(self.device)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        img = processimg(img)
        target = encoder(boxes, labels)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def _readlabel(self):

        self.box_array = []
        self.label_array = []
        for i in tqdm(range(self.length)):
            label_path = os.path.join(self.label_dir, '{}.txt'.format(self.str_(i)))
            boxes, labels = reader(label_path)
            boxes = torch.Tensor(boxes)#.to(self.device)
            labels = torch.Tensor(labels)#.to(self.device)
            self.box_array.append(boxes)
            self.label_array.append(labels)

    def _readimg(self, index):
        return cv2.imread(os.path.join(self.image_dir, '{}.jpg'.format(self.str_(index))))


## (x y x y x y x y class level) -> (7x7x26) written above
## (x y x y x y x y class level) <- (7x7x26) to be implemented in (predict)

def make_dataloader(data_dir_root, img_size=448, batch_size=16):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        ])

    trainset = YoloDataset(os.path.join(data_dir_root,'train15000'), augment=True, transform=trans)
    validset = YoloDataset(os.path.join(data_dir_root,'val1500'), augment=False, transform=trans)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return trainloader, validloader


def regroup(image, label):
    torchvision.utils.save_image(image, 'tmp.jpg')
    cv2image = cv2.imread('tmp.jpg')
    boxes, class_index, probs =  _decoder(label, 0.1)
    result = []
    for box, clsid, prob in zip(boxes, class_index, probs):
        xmin = int(float(box[0]))            
        ymin = int(float(box[1]))            
        xmax = int(float(box[2]))            
        ymax = int(float(box[3]))            
        result.append([(xmin,ymin), (xmax,ymax), clsid.item(), prob.item()])
    return cv2image, result

if __name__ == '__main__':

    # mkdir trainset trainorig trainload validset validorig validload
    ## test datafunctions with visualization

    from visualization import visualcheck, draw

    data_dir_root = 'hw2_train_val'
    # expect to call: python datafunc.py
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor(),
        ])

    trainset = YoloDataset(os.path.join(data_dir_root,'train15000'), augment=True, transform=trans)
    validset = YoloDataset(os.path.join(data_dir_root,'val1500'), augment=False, transform=trans)

    check_index = [1,3,5,12,13,111,1114,14443]

    # trainset
    for i in check_index:
        image, label = trainset[i]
        image, result = regroup(image, label)
        draw(image, result, 'tmp/trainset/%d.jpg'%i)
        visualcheck(index=i, path='tmp/trainorig/%d.jpg'%i)

    check_index = [1,3,5,12,13,111,1114,1443]

    for i in check_index:
        image, label = validset[i]
        image, result = regroup(image, label)
        draw(image, result, 'tmp/validset/%d.jpg'%i)
        visualcheck(index=i, train=False, path='tmp/validorig/%d.jpg'%i)

    trainloader, validloader = make_dataloader(data_dir_root)

    # just check 1 batch
    for images, labels in trainloader:
        for i, (image, label) in enumerate(zip(images, labels)):
            image, result = regroup(image, label)
            draw(image, result, 'tmp/trainload/%d.jpg'%i)
        break

    for images, labels in validloader:
        for i, (image, label) in enumerate(zip(images, labels)):
            image, result = regroup(image, label)
            draw(image, result, 'tmp/validload/%d.jpg'%i)
        break



    