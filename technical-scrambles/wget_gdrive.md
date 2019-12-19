
The code is  

```
  
wget --load-cookies /tmp/cookies.txt\
     "https://docs.google.com/uc?export=download&confirm=$(wget\
       --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate\
          'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pWEZsZE9oNnFzTm8' -O- |\
              sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7EVK8r0v71pWEZsZE9oNnFzTm8"\
               -O croptest.tar.gz && rm -rf /tmp/cookies.txt



```
where each `id=` should be replaced with the actual id of the file.

ref:
https://medium.com/@paudelanjanchandra/download-google-drive-files-using-wget-3c2c025a8b99
