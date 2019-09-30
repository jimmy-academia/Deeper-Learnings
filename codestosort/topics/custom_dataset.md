# How to build a custom dataset in Pytorch

> Using YOLO dataset as an example
> Based on National Taiwan University Deep Learning Computer Vision Course 2019 HW2

#### 1. Data

The data to for this example is  DOTA-v1.5, a satelite view for object detection. [source link](https://captain-whu.github.io/DOAI2019/dataset.html) (You can download the dataset by this [shell script](custom_dataset/donwload_dota_dataset.sh))  
After downloading the data, we can see that the directory is structured as below:
```
train15000
    --images
        00000.jpg .... 14999.jpg
    --labelTxt_hbb
        00000.txt .... 14999.txt

valid1500
    --images
        0000.jpg .... 1499.jpg
    --labelTxt_hbb
        0000.txt .... 1499.txt
```

#### 2. Building Pytorch Dataset

With the data at hand, we can start building our pytorch dataset! The complete code is provided [datafunc.py](custom_dataset/datafunc.py). We will go through basic ideas below

##### basics
First of all, a pytorch dataset needs ```__init__(), __getitem__(index), __len()__``` in its class defintion. One way to do this is to settle the data paths in ```__init__()``` and load the correct file in ```__getitem__()```. However loading a file or image every time might take longer (haven't experimented yet), so after some researching, I found that you can preprocess your data into torch tensors and use ```torch.load``` at ```__init__()``` to load all the needed data.  

```
import os
import torch
import torchvision
from PIL import Image

class customset(torch.utils.data.Dataset):
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

```

As can be seen above, we are yet to implement a ```create_data_file()``` for creating the 'train.pt' and 'valid.pt' for fast loading.

```
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
            . . . 
            for i in tqdm(range(total)):
                process ith image and append to image set
                process ith label and append to label set
            combine image set; label set
            torch.save((img_set, target_set), save_path)
            print('saved to', save_path)
```

The details for processing image and labels is related to yolo specifically and is omitted for clarity. Those who are interested can check the complete code in [datafunc.py](custom_dataset/datafunc.py)

#### references:
* blog post on simplified version of Mnist dataset (chinese) https://blog.csdn.net/u012436149/article/details/69061711
* pytorch torchvision MNIST dataset source code https://pytorch.org/docs/master/_modules/torchvision/datasets/mnist.html
* torch.save https://pytorch.org/docs/stable/torch.html#torch.save

### 3. Building Pytorch Dataloader

This is fairly easy, simply set up your dataset class and use ```torch.utils.data.DataLoader```
```
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
```

### sidenote
1. Following the way its done in MNIST dataset provided in torchvision, we use minimal set of torchvision.transforms in the ```self.create_data_file()``` and set up ```transform=None, target_transform=None``` variables so we can choose other transforms is later use. Therefore in ```__getitem__()```, ```img``` is preset to be a PIL image and user should always set up  ```transfroms.ToTensor()``` to convert it to torch tensors, as is the case for other torchvision datasets.
2. It seems that MNIST dataset has ByteTensor for img and LongTensor for label. The description in torchvision.transforms.ToTensor() states that PIL images with value 0-255 will be transformed to ByteTensor and 0.0-1.0 to FloaTensor. 

* torchvision.transforms https://pytorch.org/docs/stable/torchvision/transforms.html
* to check tensor type https://discuss.pytorch.org/t/print-tensor-type/3696
* transfer type with .to(type) https://discuss.pytorch.org/t/bytetensor-to-floattensor-is-slow/3672/2
