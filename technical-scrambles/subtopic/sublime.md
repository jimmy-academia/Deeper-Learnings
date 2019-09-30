# sublime
<p align="right">  
<a href="../README.md">back</a>
</p>

## skip end braket with [shift]+[space]
in sublime text -> preferences -> keybindings    
put in sublime-keymap -- User 
```
[
    {
        "keys": ["shift+space"], "command": "move", "args": {"by": "characters", "forward": true}
    }
]
```

or use [cmd] + [return] to skip to next line

## rsub: write remote file in local sublime

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




## other topic (todo)
remove pop up:
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

https://gist.github.com/egel/b7beba6f962110596660