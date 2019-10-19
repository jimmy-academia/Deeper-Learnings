# setting up a server
>after you connect to the server, set up your local device and server in the following methods to automate your work!

## ssh automation

### 1. use ssh config

create with ```touch ~/.ssh/config```
typical config file example:
```
Host server
    HostName <ip address>
    User jimmy
    Port 1000
    ServerAliveInterval 60
```

### 2. use ssh keys

in local, (if no ```~/.ssh/id_rsa``` exists):
do  ```ssh-keygen```  
then do ```ssh-copy-id username@remote_host``` and login with password  
> possible issue: key too open
> ```
> sudo chmod 600 ~/.ssh/id_rsa
> sudo chmod 600 ~/.ssh/id_rsa.pub
> ```

finally use ssh agent and save time ```ssh-add```


> ref:
(ssh key)   https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604  
(key too open issue)  https://www.howtogeek.com/168119/fixing-warning-unprotected-private-key-file-on-linux/  
(ssh agent)  https://unix.stackexchange.com/questions/12195/how-to-avoid-being-asked-passphrase-each-time-i-push-to-bitbucket   


### 3. setup .bashrc/.bash_profile in server

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


### 4. use rsub: write remote file in local sublime

#### Settings
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
ref: https://stackoverflow.com/questions/11818131/warning-remote-port-forwarding-failed-for-listen-port-52698/14594312  
haven't suceed in pip install yet, only work in source conda environment-> conda install pip-> pip install rsub

to suceed in pip install, do ```export PATH="~/.local/bin:$PATH"```
ref: https://github.com/pypa/pip/issues/3813  
