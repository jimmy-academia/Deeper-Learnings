

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wxVdrwYJv-qJtuXqZQ8jrysodOUgNj8r' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wxVdrwYJv-qJtuXqZQ8jrysodOUgNj8r" -O dota_train_val.zip && rm -rf /tmp/cookies.txt

unzip ./hw2_train_val.zip

rm ./hw2_train_val.zip
