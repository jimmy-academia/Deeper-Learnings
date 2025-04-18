
'''
variable settings and helper functions
'''
import os
import random
import numpy as np
import torch
from types import SimpleNamespace

import re
import json
import logging
import argparse


from pathlib import Path

import logging
import subprocess


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_verbose(verbose):
    # usages: logging.warning; logging.error, logging.info, logging.debug
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],  # Print to terminal
    )

# ==== read/write functions ===== #

def setup_logging(verbose=1):
    # usages: logging.warning; logging.error, logging.info, logging.debug
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],  # Print to terminal
    )

def readf(path):
    with open(path, 'r') as f:
        return f.read()

def writef(path, content):
    with open(path, 'w') as f:
        f.write(content)

def awritef(path, content):
    with open(path, 'a') as f:
        f.write(content)

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

# ==== Default Variable Setting ===== #

torch.set_printoptions(sci_mode=False)

## rootdir setting for storing dataset from pytorch
# for dir_ in ['ckpt', 'cache']:
    # os.makedirs(dir_, exist_ok=True)

default_rootdir_logpath = 'cache/rootdir'
warn_msg = f"Warning: {default_rootdir_logpath} does not exist. Setting to './cache'"
Rootdir = readf(default_rootdir_logpath) if os.path.exists(default_rootdir_logpath) else (print(warn_msg) or './cache')
Rootdir = Path(Rootdir)

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


# ==== Helper Functions ===== #

def model_to_kb(model):
    trainable_params = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    trainable_params_kb = int(trainable_params / 1024)
    return trainable_params_kb


def params_to_memory(params, precision=32):
    # Precision in bits, default is 32-bit (float32)
    bytes_per_param = precision / 8  # Convert bits to bytes
    total_bytes = params * bytes_per_param

    # Define units and corresponding thresholds
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = total_bytes
    unit_index = 0

    # Convert to the largest possible unit while size >= 1024
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    # Round the size and return the value with the appropriate unit
    return round(size), units[unit_index]

def run_command(cmd, shell=True):
    # shell = True: cmd = 'python test.py'
    # shell = False: cmd = ['python', 'test.py']
    import subprocess

    env = os.environ.copy()
    env["IS_SUBPROCESS"] = "1"

    try:
        subprocess.run(cmd, check=True, shell=shell, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")

# ==== arguments ==== #

def default_args():
    args = SimpleNamespace()
    
    # base
    args.seed = 0
    args.device = 0
    
    # main decisions
    args.model = 'resnetcifarapt'  # Default value
    args.dataset = 'cifar10'  # Default value
    args.train_env = 'AST'  # Default value
    
    # test time settings
    args.test_time = 'none'  # Default value
    args.test_domain = 'corrupt'  # Default value
    args.corrupt_level = 5  # Default value
    args.corrupt_type = 'gaussian_noise'  # Default value
    args.test_time_iter = 1  # Default value
    
    # detail train/attack decisions
    args.batch_size = 128  # Default value
    args.eps = None  # Default value
    args.attack_iters = 50  # Default value
    args.eval_interval = 1  # Default value
    
    # detail model decisions (iptnet)
    args.patch_size = 1  # Default value
    args.vocab_size = 128  # Default value
    
    # train record path or notes 
    args.ckpt = 'ckpt'  # Default value
    args.record_path = None  # Default value
    args.lr = 1e-3
    
    return args

def str2float_int_bool(value):
    if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
    try:
        return float(value) if '.' in value else int(value)
    except ValueError:
        return value

def convert_args(data):
    data = {key: str2float_int_bool(value) for key, value in data.items()}
    if 'device' in data:
        data['device'] = torch.device(data['device'])
    return data



def ensure_dir(_dir):
    """Ensure cache directory exists."""
    Path(_dir).mkdir(exist_ok=True)

def create_llm_client(keypath=".openaikey", model="gpt-4o-mini", temperature=0):
    """Create a function to call the LLM with updated OpenAI API."""
    client = OpenAI(api_key=readf(keypath).strip())
    
    def call_llm(messages, model=model, temperature=temperature, max_tokens=500):
        """Call the LLM with a prompt and return the response."""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"prompt messages: {messages}")
            return "Error processing request."
    
    return call_llm
