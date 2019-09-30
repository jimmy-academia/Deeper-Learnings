
# Compare IO speeds

## introduction

To Compare
* PIL -> PIL simd
* hdf5
* lmdb

### setting
read imgaes from the following file structure (or create duplicates in different file format) and compare the io speed.
```
data
|-train
    |- tt0027996
        |- cast
            |- tt0027996_nm0000011.jpg
            |- tt0027996_nm0051628.jpg
            |- ...
        |- candidate
            |- tt0027996_0000.jpg
            |- tt0027996_0001.jpg
            |- tt0027996_0002.jpg
            |- ...
    |- tt0074119
    |- ...
|-val
    |- ...
|-test
    |- ...
```

## experiment and results

### PIL
* PIL norm

```
total_time = 0
for __ in range(100):
    r = random.randint(0, len(indexhash))
    target_file = indexhash[r]['imgpath']
    start = time.time()
    pilimg = Image.open(target_file)
    torchimg = torchvision.transforms.ToTensor()(pilimg)
    total_time += time.time()-start

print('pil+totensor time avg 100 time', total_time/100)
```
\>\>\> 0.0149s per image

* PIL simd

\>\>\> 0.0106s per image

### hdf5
[Quick Start Guide for hdf5 and h5py](http://docs.h5py.org/en/stable/quick.html)

notes on creating `.h5` files:
1. Do not use too many groups! Will create much overhead. [ref](https://stackoverflow.com/questions/14332193/hdf5-storage-overhead) The best way is to put the every data in single dataset
2. Load large files cost more time than small files: loading an 8 image dataset costs about 0.0025s, \~1000 image dataset costs about 0.24s

n = ? load dataset of (n, 3, 224, 224) np array costs ~ 0.01s



### lmdb


