# Sublime text 3
> some sublime hacks

#### skip end braket with [shift]+[space]
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
