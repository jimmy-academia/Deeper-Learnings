# Git problems

## basics

* add file, commit files, and push files

```bash
git add .
git commit -a -m "just another commitment to greatness"
git push
```

## advanced

* new branch for fixing
```bash
git branch #list branches
git checkout -b newfix #create newfix ... commit fixes
```

* merging branches
```bash
git checkout main
git merge newfix # will fast-forward to newfix or else require manual merging
git branch --delete newfix # delete branch
```

* push new branch to github
```bash
git push -u origin new:new
```

* fetch new branch from github
```bash 
git fetch
git branch -v -a # check the branch
git switch newbranch
```

* Undo large file upload
```bash
git reset main^
```