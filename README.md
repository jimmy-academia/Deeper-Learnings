# Deeper-Learnings
my tutorials for every programing problem I have (while trying make my deep learning models work!)

## Techincal scrambles
* ### [conda cheatsheet](otherstuffs/conda-cheatsheet.pdf)
    ```bash
    
    conda create --name [name] python=3         # create new environment
    conda create --clone [old] --name [new]
    
    conda env list                              # list environments
    conda activate [name]; conda deactivate     # activate; deactivate
    conda env remove --name [name]              # delete environment
    ```
* ### [setting up a server](technical-scrambles/setup.md)
    > ssh automation; conda environment; rsub-sublime; bashrc settings
    * #### [my own fast setup](technical-scrambles/mofastsetup.md)
<!-- * ### [wget google drive files (not folder)](technical-scrambles/wget_gdrive.md) -->

* ### [git problems](technical-scrambles/gitundo.md)
    > git basics add, commit, push  
    > git branch, merge, fetch ...  
    > undo last commit for large file  

    
## Simple Python
* ### [open file](python/openfile.md)
* ### [interactive debugging](interactive_debugging.py)
> note: Visual Studio Code has better UI; for use when VScode not installed.

## Simple Pytorch
* ### [template pytorch directory](simple-pytorch/goodwork)
    > train.py; model.py; dataset.py; utils.py
* ### [Basic Usages](simple-pytorch/basic.md)
* ### [simple acceleration for Pytorch](simple-pytorch/simple_acc.md)
* ### [dataloading in Pytorch](simple-pytorch/dataloader.md)
