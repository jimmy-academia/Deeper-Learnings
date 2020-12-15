# setting up a server
>after you connect to the server, set up your local device and server in the following methods to automate your work!
>cheat sheet: [my own fast setup](technical-scrambles/mofastsetup.md)

## 1. setup ssh automation

### prepare ssh config

at local system, create the ssh config file, and write in it:  
```
touch ~/.ssh/config

    > in .ssh/config <
Host server
    HostName <ip address>
    Port <port address, if needed>
    User <user name>


    ServerAliveInterval 60 
```
### use ssh keys

```
    > if you do not have ~/.ssh/id_rsa key in local system <
ssh-keygen 
...

    > setup a ssh agent to remember your key <
eval $(ssh-agent)
ssh-add
    
    > use ssh-copy-id and login with your password <

ssh-copy-id username@remote_host
```  
> possible issue: key too open
> ```
> chmod 600 ~/.ssh/id_rsa
> chmod 600 ~/.ssh/id_rsa.pub
> chmod 600 ~/.ssh/config
> ```

> ref:
(ssh key)   https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604  
(key too open issue)  https://www.howtogeek.com/168119/fixing-warning-unprotected-private-key-file-on-linux/  
(ssh agent)  https://unix.stackexchange.com/questions/12195/how-to-avoid-being-asked-passphrase-each-time-i-push-to-bitbucket   

### 2. setup conda environment

#### Installation Guide

A. Get latest distribution from https://www.anaconda.com/distribution/, copy link and download

> check ubuntu 32 or 64 bits with `uname -a` and get
> 32 bits: `.....SMP Mon Apr 11 03:31:50 UTC 2011 i686 i686 i386 GNU/Linux`
> 64 bits: `.....SMP Wed Aug 7 18:08:02 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux` 
> 

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
```

> to check download integrity, find the hashes here: (hashes == sum) https://docs.anaconda.com/anaconda/install/hashes/all/ 
```
    > template <
echo "<known SHA 256 sum of the file> <name of the file>" | sha256sum -c
    > example <
echo "2b9f088b2022edb474915d9f69a803d6449d5fdb4c303041f60ac4aefcc208bb Anaconda3-2020.02-Linux-x86_64.sh" | sha256sum -c            
```

B. Run script to install
`bash Anaconda3-2020.02-Linux-x86_64.sh`
(say yes to prepend in bashrc or run init!)

C. Activate, test and create new Environment
```
source ~/.bashrc
conda list
conda create --name my_env python=3
conda activate my_env
```

#### Usage
for usage see [conda cheatsheet](https://github.com/jimmy-academia/Deeper-Learnings/blob/master/otherstuffs/conda_cheatsheet.jpeg)

>ref: 
(check server 64 or 32) https://askubuntu.com/questions/41332/how-do-i-check-if-i-have-a-32-bit-or-a-64-bit-os
(download integrity) https://superuser.com/questions/1312740/how-to-take-sha256sum-of-file-and-compare-to-check-in-one-line
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart


### 3. Install rsub: write remote file in local sublime text editor

#### Installation Guide
A. Install rsub on server
```
conda install pip
pip install rsub
```

B. Install rsub on local sublime3 
(Sublime Text 3) Open Package Manager (
```
Ctrl-Shift-P   Linux/Windows,
Cmd-Shift-P    Mac,
    press `Install Package`
    search and press `rsub`
```
    
C. Configure .ssh/config with port forwarding
```
Host <ssh-name>
    HostName url.com
    Port <port number>
    User <user name>
    RemoteForward 52698 localhost:52698 (for rsub)
    ServerAliveInterval 60
    ForwardAgent yes  (ssh agent after exit from middle server, this is better)
```
> or use command line with port forwarding
`ssh -R 52698:localhost:52698 server_user@server_address`


D. Finish
`rsub path_to_file/file.txt` in server opens the file in local sublime text 3


> main ref:
https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3
ref:
http://log.liminastudio.com/writing/tutorials/sublime-tunnel-of-love-how-to-edit-remote-files-with-sublime-text-via-an-ssh-tunnel  
http://blog.keyrus.co.uk/editing_files_on_a_remote_server_using_sublime.html

#### Cautions and debuggings
will fail when there is another ssh session to same server!  

fix: change to a different port (ex: 52699) and fix in the rsub main.py
```
    > shows path to rsub (it is a python executable file) <
which rsub
    
    > open the rsub file (use vim!!, using rsub will break the executable and turn it to plain text) <
        
        find:

        # #########################  M a i n  ######################## #

        def main():
            # Defaults
            conf = AttrDict({'host': 'localhost', 'port': 52698})

        and change it to e.g. 52699
    
        than change your config (in local) to:
            RemoteForward 52699 localhost:52698
```

For secondary connection, connect the second server to the port that links to local at the first server, e.g., in above case, write `RemoteForward 52700 localhost:52699` in server 1  
so now `server 2 port 52700 -> server 1 port 52699 -> local port 52698 -> sublime text 3`

If secondary connection has fixed names, try the following
```
function ssh {
    if [ "$1" == "comp" ]; then
        command ssh -R 52699:localhost:52699 comp
    elif [ "$1" == "comp2" ]; then
        command ssh -R 52699:localhost:52699 comp2
    else
        command ssh $1
    fi
}
# two [[   ]] causes verbose [[lab not found, only 1[], need spaces!
```

> ref: https://stackoverflow.com/questions/11818131/warning-remote-port-forwarding-failed-for-listen-port-52698/14594312  
to suceed in pip install, do `export PATH="~/.local/bin:$PATH"`
ref: https://github.com/pypa/pip/issues/3813  

### 3.5 useful sublime preferance settings
`Preferences/Setting`
```
{
	"ignored_packages":
	[
		"Vintage"
	],
	"open_files_in_new_window": true,
	"translate_tabs_to_spaces": true,
	"word_wrap": true,
}
```
`Preferences/Key Bindings`
```
[
	{
		"keys": ["shift+space"], "command": "move", "args": {"by": "characters", "forward": true}
	}	
]
```

### 4. Setup .bashrc/.bash_profile for proper greeting in server

some functions I add in `~/.bashrc` (after `conda init` except first and second part)

```
## at the very top: 
# if not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac


## put on top of conda for colorful command prompt
export PS1="\[\033[38;5;39m\]\u\[\033[90m\] at \[\033[32m\]\h \[\033[90m\]in \[\033[33m\]\w\[\033[m\]$ "



source activate <env name>
alias subl='rsub'
alias gpuu='watch -n 0.1 nvidia-smi'

function hello(){
	echo "what a wonderful day!"
	echo "let's start working!!"
    cd <some work directory>
    ls
}

or even better (with color):

function hello(){
    echo "+ + + + + + + + + + + + + + + + + + + + + + + + "
    echo ""
    echo -e "\e[93m  Greeting Message \e[0m"
    echo ""
    echo "+ + + + + + + + + + + + + + + + + + + + + + + + "
    echo -e "\e[38;5;208m  taking you to ~/house \e[0m"
    echo -e "\e[38;5;208m  starting location for current projects \e[0m"
}

function dogit() {
    git add .
    git commit -a -m "$1"
    git push
}

hello
```

#### WARNING: the first is needed if you want to use the hello function without breaking scp

>ref:  
(how bashrc work)https://www.thegeekstuff.com/2008/10/execution-sequence-for-bash_profile-bashrc-bash_login-profile-and-bash_logout/
(color code, for PS1)https://gist.github.com/jbutton/9874192  
(color code, for echo -e)https://misc.flogisoft.com/bash/tip_colors_and_formatting  
(scp breaks)https://superuser.com/questions/395356/scp-doesnt-work-but-ssh-does
https://superuser.com/questions/395356/scp-doesnt-work-but-ssh-does/1380194#1380194

>sidenote
editing .bashrc have no effect at first cause the server I ssh into
is actually using zsh, took me a while to realize.
simply adding ```source .bashrc``` in zshrc finished it all.

### 5. Miscellaneous
* setup screen white bar (to differentiate screen and ssh mode)
```
touch ~/.screenrc

    > in ~/.screenrc <
hardstatus on
hardstatus alwayslastline
hardstatus string "%S"
```

* hush login prompt (Welcome to Ubuntu) for Ubuntu machines:
```touch ~/.hushlogin```
https://askubuntu.com/questions/676374/how-to-disable-welcome-message-after-ssh-login
and write 
```last -w | grep "$USER" | head -n1 | perl -lane 'END{print "Last login: @F[3..6] $F[8] from $F[2]"}'```
in `.ssh/rc` to retain login information
NOTE: add the interactive clause (4.) to prevent breaking scp 
https://unix.stackexchange.com/questions/260813/bash-hushlogin-keep-last-login-time-and-host

* bash script for downloading from google drive
```
function gdown() {
    # gdown <FILE_ID> <OUTPUT_FILENAME>
    file_id=$1
    file_name=$2
    
    # first stage to get the warning html
    curl -c /tmp/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id" > \
    /tmp/intermezzo.html

    # second stage to extract the download link from html above
    download_link=$(cat /tmp/intermezzo.html | \
    grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
    sed 's/\&amp;/\&/g')
    curl -L -b /tmp/cookies \
    "https://drive.google.com$download_link" > $file_name
}
```
>ref:
https://afun.medium.com/downloading-big-file-form-google-drive-with-curl-7918bc3b2605
