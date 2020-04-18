# setting up a server
>after you connect to the server, set up your local device and server in the following methods to automate your work!

## 1. ssh automation

### prepare ssh config

at local system, create the ssh config file, and write in it:  
```
touch ~/.ssh/config

    Host server
        HostName <ip address>
        User jimmy
        Port 1000
        ServerAliveInterval 60
```
### use ssh keys

```
    >if you do not have ~/.ssh/id_rsa key in local system<
ssh-keygen 
...

    >setup a ssh agent to remember your key<
eval $(ssh-agent)
ssh-add
    
    >use ssh-copy-id and login with your password<

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

### 2. install conda environment

#### Installing Guide
A. get latest distribution from https://www.anaconda.com/distribution/
> check ubuntu 32 or 64 bits
> ```uname -a```
> 32 bits: ```.....SMP Mon Apr 11 03:31:50 UTC 2011 i686 i686 i386 GNU/Linux```
> 64 bits: ```.....SMP Wed Aug 7 18:08:02 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux```
> ref: https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

B. copy download link and download the script

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
```
(and maybe check integrity [ref](https://superuser.com/questions/1312740/how-to-take-sha256sum-of-file-and-compare-to-check-in-one-line))
find the hashes here: https://docs.anaconda.com/anaconda/install/hashes/all/ 
template
```
echo "<known SHA 256 sum of the file> <name of the file>" | sha256sum -c
```
example
```
echo "2b9f088b2022edb474915d9f69a803d6449d5fdb4c303041f60ac4aefcc208bb Anaconda3-2020.02-Linux-x86_64.sh" | sha256sum -c
```

C. run install
`bash Anaconda3-2020.02-Linux-x86_64.sh`

(say yes to prepend in bashrc or run init!)

D. Activate, Test and create new Environment
```
source ~/.bashrc
conda list
conda create --name my_env python=3
conda activate my_env
```

ref: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

### 4. use rsub: write remote file in local sublime

#### Installing Guide
A. install rsub on server
(can use conda install pip; pip install rsub for package control)

B. install rsub on sublime3
    On Sublime Text 3, open Package Manager (Ctrl-Shift-P on Linux/Win, Cmd-Shift-P on Mac, Install Package), 
    search for rsub and install it

C. Open command line with port forwarding  
```ssh -R 52698:localhost:52698 server_user@server_address```
or configure .ssh/config 
```
Host <ssh-name>
 	HostName url.com
	Port <port number>
	User <user name>
	RemoteForward 52698 localhost:52698
```
then do ```ssh <ssh name>```  

D. Finish
```rsub path_to_file/file.txt```
File opening auto in Sublime 3

main ref:
https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3

ref:
http://log.liminastudio.com/writing/tutorials/sublime-tunnel-of-love-how-to-edit-remote-files-with-sublime-text-via-an-ssh-tunnel  
http://blog.keyrus.co.uk/editing_files_on_a_remote_server_using_sublime.html

#### Cautions and debuggings
will fail when there is another ssh session to same server!  

fix: change to a different port (ex: 52699) and fix in the rsub main.py
```
RemoteForward 52699 localhost:52698
```

ref: https://stackoverflow.com/questions/11818131/warning-remote-port-forwarding-failed-for-listen-port-52698/14594312  
haven't suceed in pip install yet, only work in source conda environment-> conda install pip-> pip install rsub

to suceed in pip install, do ```export PATH="~/.local/bin:$PATH"```
ref: https://github.com/pypa/pip/issues/3813  


### 5. setup .bashrc/.bash_profile in server

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

> Cautions and debuggings
editing .bashrc have no effect at first cause the server I ssh into
is actually using zsh, took me a while to realize.
simply adding ```source .bashrc``` in zshrc finished it all.


