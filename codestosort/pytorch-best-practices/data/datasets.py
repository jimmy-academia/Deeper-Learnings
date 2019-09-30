
from torchvision import datasets
from torchvision import transforms

def MNIST(data_dir_root, train):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    dataset = datasets.MNIST(
        data_dir_root+'/mnist', train=train, download=True, transform=trans
    )
    return dataset

def CIFAR10(data_dir_root, train):
    trans = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

	# load dataset
	dataset = datasets.CIFAR10(
	    data_dir_root+'/cifar10/', train=train, download=True, transform=trans
	)
	return dataset

def CIFAR100(data_dir_root, train):
    trans = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

	# load dataset
	dataset = datasets.CIFAR100(
	    data_dir_root+'/cifar100/', train=train, download=True, transform=trans
	)
	return dataset

	





# import numpy as np 
# np.random.seed(322)
import torch.utils.data as data
from PIL import Image
import os


class MiniImagenet(data.Dataset):
    def __init__(self):        
        self.root_path = '../../dataset/mini/train/n01532829/'
        self.data = readfile(self.root_path)
        self.transforms = transforms.Compose([
            file_PIL, 
            transforms.CenterCrop(224), 
            transforms.Resize(256),  #was 84, 64  256 for large
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
            ])
        self.n_episodes = len(self.data)
        # self.generate_episodes(self.n_episodes)

    # def generate_episodes(self, n_episodes):
    #     self.batchset = []
    #     for _ in range(n_episodes):
    #         imgs = np.random.choice(self.data, 1, False)
    #         self.batchset.append(imgs)

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        return self.transforms(self.data[index])

        # nc = 3
        # sz = 84
        # batchset = torch.FloatTensor(self.batchsize, nc, sz, sz)
        # for n,sample_path in enumerate(self.batchset[index]):
        #     batchset[n] = self.transforms(sample_path)
        # return batchset

def readfile(path):     # path should have / in the end
    img_list = os.listdir(path)
    file_list = []
    _tensor = transforms.ToTensor()
    for file in img_list:
        img_tensor = _tensor(Image.open(path+file))
        if img_tensor.shape[0] == 3:
            file_list.append(path+file)
    return file_list

file_PIL = lambda x: Image.open(x)