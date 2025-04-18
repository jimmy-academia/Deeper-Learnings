# Basic Usages

## argparse
```
import argparse

parser = argparse.ArgumentParser(description='a new parser')
parser.add_argument('--args', type=str, default='default', required=True, help='args')
args = parser.parse_args()
```

### advanced, save and load arguments
```
import os
import sys
import argparse

import torch


def handle_arguments():
    parser = argparse.ArgumentParser(description='init taskname, imagepath, modelpath; info will be stored')
    parser.add_argument('--taskname', type=str, help='taskname', required=True)
    parser.add_argument('--checkpoints_dir', type=str, default='checkfiles')
    parser.add_argument('--imagepath', type=str, help='imagepath')
    parser.add_argument('--modelpath', type=str, help='modelpath')
    parser.add_argument('--imgside', type=int, default=256, help='input image size ex: 256')
    parser.add_argument('--sqside', type=int, default=32, help='size of jacobpiece ex: 32')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.checkpoints_dir, args.taskname)):
        os.makedirs(os.path.join(args.checkpoints_dir, args.taskname))

    # load options
    option_path = os.path.join(args.checkpoints_dir, args.taskname, 'opt.pth')
    init_list = [args.imagepath, args.modelpath]
    if not os.path.exists(option_path):
        if None in init_list:
            print('REQUIRED:', '[args.imagepath, args.modelpath]')
            print('RECEIVED:', init_list)
            print('WARNING! some required arguments are not initialized')
            sys.exit(0)
        else:
            torch.save(args, option_path) 
    else:
        args = torch.load(option_path)

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in vars(args).items():
        comment = ''
        default = parser.get_default(k)
        # if v != default:
            # comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    return args
```


## directory create if not exist
```
import os
for dir_ in ['dir_path', 'dir_path2'...]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)
```


