# Best Practice for dataloading in Pytorch

## 1. The basic way

```
import torch
class Dataset(torch.utils.data.Dataset):

    def __init__(self):

        super(Dataset, self).__init__()
        ... initialize ...

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ... get items for 1 index ...
        return items

dataset = Dataset()
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            )
```

### notes
num_workers set to > 0 creates new processes to load from dataset and can run parallely while GPU is training model[[1]](Reference)   
pin_memory helps when you want to move data to GPU to train[[2]](References)   

## 2. Improvements

### faster IO:
* PIL -> PIL simd
* hdf5
* lmdb

### CPU acceleration
* PIL -> PIL simd (for image transformations)

main page: [speed comparison](speedcompare.md)

### GPU acceleration
* NVIDIA DALI

main page: [NVIDIA DALI](nvidia_dali.md)  

For installation method check [installation](install.md)


## References
1. https://blog.csdn.net/u014380165/article/details/79058479
2. https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

