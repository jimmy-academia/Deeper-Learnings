# Simple Acceleration Methods

## dataloading

### 1. replace PIL with PIL simd

> the following is from [fast.ai](https://docs.fast.ai/performance.html):   

#### Installation:
```
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install -v --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

#### Confirmation
Check if python is using pillow-simd (`x.x.x.postx` indicates simd)
``` 
python -c "from PIL import Image; print(Image.PILLOW_VERSION)"
```
Check if libjpeg-turbo replaced libjpeg:
```
cd ~/anaconda3/envs/<env-name>/lib/python<version>/site-packages/PIL/
ldd  _imaging.cpython-<36>m-x86_64-linux-gnu.so | grep libjpeg
```
> number 36 may be different, press tab after `_imaging.` should lead to correct file

possible result: `libjpeg.so.8 => ~/anaconda3/envs/<env-name>/lib/libjpeg.so.8`

Than check 
```
ls ~/anaconda3/env/<env-name>/conda-meta/ |grep libjpeg
```
to find `libjpeg-turbo-x.x......json`

Finnaly check if pillow-simd is using turbo
```
python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
```
to find `True`

### 2. use these cuda lines
```
img = img.cuda(non_blocking=True)
```
and 
```
if __name__ == '__main__':
    import torch
    torch.backends.cudnn.benchmark = True     
    torch.backends.cudnn.enabled = True
```