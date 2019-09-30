# basic pipeline

<p align="right">  
<a href="README.md">back</a>
</p>
create 3 files: datafunc.py, model.py, trainer.py  

### datafunc
> in datafunc.py, create functions for generating dataset and dataloaders    
for example:
```
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

def MNIST(data_dir_root, img_size):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(
        data_dir_root, train=True, download=False, transform=trans
    )
    return dataset

def make_dataloader(data_dir_root, img_size, batch_size):

    trainset = MNIST(data_dir_root, img_size)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader
```

### model.py
> create model class, which includes save and load options   
for example:
```
class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.G = Generator(args)
        self.G.apply(weights_init)
        self.D = Discriminator(args)
        self.D.apply(weights_init)

    def save(self, filepath):
        state = {
            'gen_net': self.G.state_dict(),
            'dis_net': self.D.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.G.load_state_dict(state['gen_net'])
        self.D.load_state_dict(state['dis_net'])
```
### trainer.py
> create trainer class that loads dataset and models, and perform training    
for example:
```
class Trainer():
    def __init__(self, config, args, opt):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = VGG().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        self.criterion = nn.BCELoss()

        self.dataloader = make_dataloader()

    def train(self):
        for i in range(self.opt.epochs):
            epoch_records = self.train_one_epoch()
            self.model.save()

    def train_one_epoch(self):
        pbar = tqdm(self.dataloader)
        for batch, label in pbar:
            out = self.model(batch)
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()            
            loss.backward()
            self.optimizer.step()
            message = 'message'
            pbar.set_description(message)
    def test(self):
        pass
```

### references:
https://github.com/kevinzakka/recurrent-visual-attention
