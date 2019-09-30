# Terminal
<p align="right">  
<a href="../README.md">back</a>
</p>

table of content    
> [basics](#basics)   
> [screenrc](#screenrc)     
> [disc space](#disc-space)    
> [ssh](#ssh)    
> [bashrc](#bashrc)
> [other](#other)

## Basics
```ls``` | list file/ ```ls -a``` for hidden (ex: .ssh/, .bash_profile)  
```cd``` | move to folder ( ```cd ..``` for parent folder)  
```mkdir``` | create a folder  
```touch``` | create a file  

```scp file user@remotehost```
ref: https://www.hypexr.org/linux_scp_help.php 

## screenrc

#### common operations

list: ```screen -ls``` <br>
create: ```screen -S <name>``` <br>
leave screen: press [ctrl]+[a]+[d] <br>
scroll cursor press [ctrl]+[a]+[esc] <br>
kill screen session: ```exit``` within screen session or do ```kill <pid>``` (find pid in ```screen -ls```)<br>

#### create session name on bottom to show that terminal is in a screen session
in ~/.screenrc
```
hardstatus on
hardstatus alwayslastline
hardstatus string "%S"
```
ref: [stackoverflow](https://stackoverflow.com/questions/2479683/how-do-i-display-the-current-session-name-or-sockname-of-a-screen-session-in)

## disc space
check free space

```df -h```

list usage sorted

```du -hs * | sort -h```

ref:
https://www.tecmint.com/how-to-check-disk-space-in-linux/
https://serverfault.com/questions/62411/how-can-i-sort-du-h-output-by-size


## ssh
basic usage:
ssh username@userip

create ssh key for ssh into remote server: ```ssh-keygen```; ```ssh-copy-id username@remote_host```  
issue: key too open
```
sudo chmod 600 ~/.ssh/id_rsa
sudo chmod 600 ~/.ssh/id_rsa.pub
```

use ssh agent and save time
```ssh-add```

ref:
(ssh key)   https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604  
(key too open issue)  https://www.howtogeek.com/168119/fixing-warning-unprotected-private-key-file-on-linux/  
(ssh agent)  https://unix.stackexchange.com/questions/12195/how-to-avoid-being-asked-passphrase-each-time-i-push-to-bitbucket   


## bashrc 
> .bashrc/.bash_profile

some functions I add in .bashrc

```
source activate machine
alias subl='rsub'

function hello(){
    echo "what a wonderful day!"
    echo "let's start working!!"
}
hello

function dogit() {
    git add .
    git commit -a -m "$1"
    git push
}
```

first 2 lines are for rsub (see below)
last function is for git

ref:  
(how bashrc work)https://www.thegeekstuff.com/2008/10/execution-sequence-for-bash_profile-bashrc-bash_login-profile-and-bash_logout/

#### Cautions and debuggings
editing .bashrc have no effect at first cause the server I ssh into
is actually using zsh, took me a while to realize.
simply adding ```source .bashrc``` in zshrc finished it all.


## other  
use gdown   
pip install gdown  
gdown \<google drive link\>  

source:  
https://github.com/wkentaro/gdown