
source ~/.bash_cdd

# scd [cdd] path/to/cdd
# scd remove cdd
# scd ls

function scd() {
    local alias_name="$1"
    local dir_path="$2"

    # Check if the action is to list aliases
    if [ -z "$alias_name" ] || [ "$alias_name" = "help" ] || [ "$alias_name" = "-h" ]; then
        # scd ls
        echo 
        echo "[Help function] for setting cdd aliases"
        echo " - 'scd [-h]/[help]' for this help message"
        echo " - 'scd ls' to list aliases"
        echo " - 'scd cd? ?/relative/path/? to set alias'"
        echo " - 'scd [rm/remove] (cdd) to remove alias'"
        echo 
        return
    fi
    if [ "$alias_name" = "ls" ] || [ "$alias_name" = "list" ]; then
        echo "Current aliases in ~/.bash_cdd:"
        echo
        cat ~/.bash_cdd
        echo
        return
    fi

    # Check if the action is to remove an alias
    if [ "$alias_name" = "rm" ] || [ "$alias_name" = "remove" ]; then
        # Check if an alias name is provided
        if [ -z "$dir_path" ]; then
            echo "Usage: scd remove alias_name"
            return
        fi

        # Get the alias line to be removed
        local alias_line
        alias_line=$(grep "^alias $dir_path=" ~/.bash_cdd)

        if [ -n "$alias_line" ]; then
            # Remove the alias line from ~/.bash_cdd
            sed -i "/^alias $dir_path=/d" ~/.bash_cdd

            echo "Removed $alias_line"
        else
            echo "Alias not found: $dir_path"
        fi

        return
    fi

    # If only one argument is provided, treat it as the path and use the default alias 'cdd'
    if [ -z "$dir_path" ]; then
        dir_path="$alias_name"
        alias_name="cdd"
    fi

    # Get the absolute path
    local abs_path
    abs_path=$(realpath "$dir_path")

    local alias_line="alias $alias_name='cd $abs_path'"

    # Check if the alias already exists in the file
    if grep -q "^alias $alias_name=" ~/.bash_cdd; then
        # Update the existing alias line
        sed -i "s|^alias $alias_name=.*|$alias_line|" ~/.bash_cdd
    else
        # Append the new alias line to ~/.bash_cd_aliases
        echo "$alias_line" >> ~/.bash_cdd
    fi

    # Source the updated file to apply the changes immediately in the current shell session
    source ~/.bash_cdd

    echo "$alias_name set to 'cd $abs_path'"
}


# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

alias tt='screen'
# tmux
alias tn='tmux new-session -s'
alias ta='tmux attach-session -t'
alias tl='tmux list-sessions'

alias dff='df -h ./'
alias duu='du -ah --max-depth=1 ./ | sort -hr'
alias sorc='source ~/.bashrc'
alias souv='source .venv/bin/activate'

alias python='python3'


# function pssu() {
#     username=jimmyyeh  # Username as the first argument
#     port=52698      # Port number as the second argument (optional, defaults to 52698)
#     pids=$(ps -u "$username" | grep sshd | awk '{print $1}')
#     for pid in $pids; do
#     lsof -i :"$port" -p "$pid" > /dev/null 2>&1
#     if [ $? -eq 0 ]; then
#       echo "PID $pid is using port $port"
#     fi
#     done
# }
alias pssu='ps -u jimmyyeh |grep sshd'
alias gpuu='watch -n 0.1 nvidia-smi'

function supertranspose() {
    awk '
        { 
            for (i=1; i<=NF; i++)  {
                a[NR,i] = $i
            }
        }
        NF>p { p = NF }
        END {    
            rowname["1"] = "the GPU id:\t";
            rowname["2"] = "used memory:\t";
            for(j=1; j<=p; j++) {
                if (a[1,2] <= 200)
                    str=rowname[j]"\033[1;35m"a[1,j]"\033[37m"
                else
                    str=rowname[j]a[1,j]
                for(i=2; i<=NR; i++){
                    if (a[i,2] <= 200)
                        str=str"\t\033[1;35m"a[i,j]"\033[37m";
                    else
                        str=str"\t"a[i,j];
                }
                print str
            }
        }' $1
}

function gg(){
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits,noheader | awk '{print substr($1, 1, 1) " " $3-$2}' | sort -n -k1.3 | supertranspose
}


LS_COLORS=$LS_COLORS:'di=36:*.py=0;37:' ; export LS_COLORS

function parse_git_branch() {
    echo | git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ [\1]/'
}
function parse_conda_env() {
    echo | conda env list | grep '*' | awk 'END {print "("$1")"}'    
}
function file_count(){
    echo | ls | wc -l
}

function cd {
    DIR="$*";
        # if no DIR given, go home
        if [ $# -lt 1 ]; then
                DIR=$HOME;
    fi;
    builtin cd "${DIR}" &&\
    # export PS1="\[\033[31m\]\u\[\033[90m\] at \[\033[32m\]\h \[\033[90m\]in \[\033[33m\]\w\[\033[1;90m\]$(parse_git_branch)\[\033[00m\] $ " &&\
    export PS1="\[\033[31m\]\u\[\033[90m\] at \[\033[32m\]\h \[\033[90m\]in \[\033[33m\]\w\[\033[1;90m\]$parse_git_branch\[\033[00m\] $ " &&\
    # export PS1="$(parse_conda_env) \[\033[31m\]\u\[\033[90m\] at \[\033[32m\]\h \[\033[90m\]in \[\033[33m\]\w\[\033[1;90m\]$parse_git_branch\[\033[00m\] $ " &&\
    if [ $(file_count) -gt 30 ]; then
        ls -b | head -30 | xargs ls -F --color=auto
        echo -e "\e[90m......total $(file_count) files/directories \e[0m" 
    else
        ls -F --color=auto
    fi
    }

PROMPT_DIRTRIM=2

function hello(){
    echo -e "\e[90m+ + + + + + + + + + + + + + + + + + + + + + + + cuda"
    echo ""
    echo -e "\e[1;31m  This is\e[38;5;208m the day\e[33m that THE LORD\e[32m has made"
    echo -e "\e[38;5;64m    And we'll\e[96m rejoice and\e[34m be glad\e[35m in it!!\e[0m\e[90m"    
    echo ""
    echo "+ + + + + + + + + + + + + + + + + + + + + + + + 12.0"
    echo -e "\e[38;5;208m  taking you to ~/Documents/CRATER \e[0m"
    echo -e "\e[38;5;208m  starting location for current projects \e[0m"
    cdd
}

function dogit() {
    git add .
    if [ $# -eq 0 ]
        then
            echo "No arguments supplied"
            git commit -a -m "just another commitment to greatness"
        else
            git commit -a -m "$1"
    fi
    git push
}

function domerge(){
    echo "a function for merging to main branch"
}

export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
