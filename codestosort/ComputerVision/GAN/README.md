# Dense Convolutional Generative Adversarial Network (DCGAN)

sample codes in [module](module)


### Errorlogs

> Code couldn't produce correct result, found on Github: [DCGAN for MNIST](https://github.com/togheppi/DCGAN/blob/master/MNIST_DCGAN_pytorch.py) and made following modifications to match the template. 

**Main problem**   
```One extra batchnorm layer at the end of D!```

**other modifications**  
§ added bias for Conv2d, ConvTranspose2d § relu set inplace=False
§ weight init for bias (for Conv)
§ model size could be too small
```self.layer_G = [(64,7,1,0), (32,4,2,1), (1,4,2,1)]
self.layer_D = [(32,4,2,1), (64,4,2,1), (1,7,1,0)]
to 
self.layer_G = [(1024,4,1,0), (512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,2,1)]
self.layer_D = [(1024,4,2,1), (512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,1,0)]
to
self.layer_D = [(128,4,2,1), (256,4,2,1), (512,4,2,1), (1024,4,2,1), (1,4,1,0)]
```   
§ fake = 0, real = 1 vs. fake=1, real=0   
§ learning rate = 0.0002 vs 0.0003   
§ errD_real + errD_fake = error, error.backwards vs errD_real.backwards, errD_fake.backwards   
§ D zerograd when training G   

**more description**  
After removing batchnorm layer at the end, `D(x) ~ 1 and D(G(z))~0` while training, indicating a optimal D at every step. It therefore seems that layersize/learning rate etc. can be arbitrary set, but batchnorm layer in critical step (esp final layer) can disrupt the whole thing!
Proper weight init increase speed of Generator to learn, and can prevent D overpowering G and cause zero gradient. This still happens from time to time.

---

> naming python script as copy.py cause import errors. (import torch, import numpy) within the file would go wrong, and afterwards doing import torch, import numpy would still spit errors
Came upon messages:
```
>>> import torch   
Segmentation fault (core dumped)

AttributeError: type object 'torch._C.FloatTensorBase' has no attribute 'numpy' 
```
could be related to   
https://github.com/python/cpython/blob/master/Lib/copy.py  
https://docs.python.org/3/library/copy.html  
search term [shallow and deep copy]

Ans: rename the script and problem is cleared
