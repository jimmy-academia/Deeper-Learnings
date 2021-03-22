import torch
import torchvision
from PIL import Image

Image_open = lambda x: Image.open(x)

class imageset(torch.data.Dataset):
    def __init__(self):        
        self.image_list = <prepare a list of path of images....>
        self.transforms = transforms.Compose([
            Image_open, 
            transforms.Resize(256),
            transforms.CenterCrop(256), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
            ])
        self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.transforms(self.data[index])

def make_loader():
    dset = imageset(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dset, 
        batch_size=batch_size,
        num_workers=8, 
        pin_memory=True,
        shuffle=False,
    )
    return data_loader
