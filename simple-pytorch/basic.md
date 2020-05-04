# Basic Usages

## argparse
```
import argparse

parser = argparse.ArgumentParser(description='a new parser')
parser.add_argument('--args', type=str, default='default', required=True, help='args')
```

