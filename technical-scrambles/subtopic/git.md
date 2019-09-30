# Git technical issues
<p align="right">  
<a href="../README.md">back</a>
</p>
table of content  
> [gitignore](#gitignore)     
> [large file](#large-file)    
> [write latex](#write-latex-in-git-markdown)    

## gitignore    

create .gitignore file and place for example:
```
**/__pycache__
```
ref:  
[gitignore](https://www.atlassian.com/git/tutorials/saving-changes/gitignore)
  
to remove files already checked into git  
```
git rm --cached FILENAME
git rm --cached -r DIRNAME
```
## large file
```
git reset --soft HEAD~3 (3 is number of commits followed by failed push)
git commit -m "New message for the combined commit"
```
ref:  
[can't push to github because of large file](https://stackoverflow.com/questions/19573031/cant-push-to-github-because-of-large-file-which-i-already-deleted)    
[gitignore by filesize?](https://stackoverflow.com/questions/4035779/gitignore-by-file-size)  

## write latex in Git markdown:
I recommend [TeXify](https://github.com/settings/installations/680004)  
Install on github and select repositries to use. It will check in the repositries for \*.tex.md and convert all tex notations within into SVG images to create \*.md file (as well as folder for the images tex/\*)  
Small bug is that it currently won't covert latex in nested list.