# RSUB: the best way to edit files on server (if you don't knw VIM)

> with rsub, you can write remote files in local sublime editor

### Installation:
1) On Server: `pip install rsub`
2) On Sublime in your computer install Package Control:
    * open package manager `Ctrl-Shift-P` on Linux/Win, `Cmd-Shift-P` on Mac, search and click `Package Control: Install Package`
    * wait for search bar to apear, search and install rsub
3) Now ssh to remote server with port forwarding:
```ssh -R 52698:localhost:52698 server_user@server_address```
or configure .ssh/config 
```
Host <server-name>
    HostName <url.com/ip-adress>
    Port <port number>
    User <user name>
    RemoteForward 52698 localhost:52698
```
then do ```ssh <server-name>```  

now `rsub <file>` on server will open up file in local sublime text

p.s. I put `alias subl='rsub'` in my .bashrc for convenience

### Caution

Remote port forwarding will fail when there is another ssh session to same server! This may happen because 
1. another user is also using rsub
2. Your have another ssh connection at the same time
3. Your previous ssh connection did not close of properly

For 1. Find your rsub file (`which rsub`) and edit the file located at the path. Find 
```
def main():
    # Defaults
    conf = AttrDict({'host': 'localhost', 'port': 52698})
```
in the file and change the port from 52698 to other numbers (e.g. 52699), and also configure your ssh config to `RemoteForward 52699 localhost:52698`

For 2.
If you are have a ssh connection on one terminal with remote port forwarding and want to open another ssh connection, you can still use rsub on the second terminal and it will work (since you got the connection, just via the first terminal).

For 3. 
us `ps -u <username>` to find previous sshd processes and kill it. Then exit and reconnect again and it should work.

If pip install won't work, do ```export PATH="~/.local/bin:$PATH"```

### References

https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3
http://log.liminastudio.com/writing/tutorials/sublime-tunnel-of-love-how-to-edit-remote-files-with-sublime-text-via-an-ssh-tunnel  
http://blog.keyrus.co.uk/editing_files_on_a_remote_server_using_sublime.html
https://stackoverflow.com/questions/11818131/warning-remote-port-forwarding-failed-for-listen-port-52698/14594312  
https://github.com/pypa/pip/issues/3813  


<!-- ## future topic ?
remove pop up?
https://forum.sublimetext.com/t/solved-disable-update-available-pop-up/1381
```
{
    "font_size": 9,
    "update_check": false,
    "ignored_packages":
    [
        "Vintage"
    ],
    "tab_size":4,
    "translate_tabs_to_spaces": true
}
```

https://gist.github.com/egel/b7beba6f962110596660 -->