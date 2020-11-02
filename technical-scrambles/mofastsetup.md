# my own fast setup

### ssh
make `.ssh/config` file
```
ssh-copy-id "servername"
```

### conda
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh  [yes to bash init!]
source ~/.bashrc
conda create --name quiver python=3
conda activate quiver
pip install rsub
vim anaconda3/.../rsub    [which rsub to be sure][/52698 -> 52699; avoid conflict]
```

### bashrc
```
rsub ~/.bashrc
```
##### add lines above conda

`export PS1="\[\033[38;5;39m\]\u\[\033[90m\] at \[\033[32m\]\h \[\033[90m\]in \[\033[33m\]\w\[\033[m\]$ "`

##### add lines bottom

```


source activate quiver
alias subl='rsub'
alias gpuu='watch -n 0.1 nvidia-smi'

function hello(){
    echo "+ + + + + + + + + + + + + + + + + + + + + + + + "
    echo ""
    echo -e "\e[93m  Greeting Message \e[0m"
    echo ""
    echo "+ + + + + + + + + + + + + + + + + + + + + + + + "
    echo -e "\e[38;5;208m  taking you to ~/house \e[0m"
    echo -e "\e[38;5;208m  starting location for current projects \e[0m"
    echo ""
}

function dogit() {
    git add .
    git commit -a -m "$1"
    git push
}

hello
```
### final
```
subl ~/.screenrc

hardstatus on
hardstatus alwayslastline
hardstatus string "%S"

touch ~/.hushlogin
```