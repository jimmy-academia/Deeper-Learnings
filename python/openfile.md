# How to open file and read lines

basic structure
```python
with open(filepath, 'r') as f:
    f.readline()    # remove first line
    for line in f:
        # do things with oneline

```