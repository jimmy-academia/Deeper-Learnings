

from model import GoodModel
from dataset import make_loader
from utils import make_define_dir
import argparse


parser = argparse.ArgumentParser(description='a great new task')
parser.add_argument('--taskname', type=str, required=True)
parser.add_argument('--gid', type=int, default=0)
opt = parser.parse_args()

class arguments:
    def __init__(self, args): 
        self.reslt_dir = 'results'
        self.taskname = args.taskname
        self.task_dir = os.path.join(self.reslt_dir, self.taskname)
        self.result_path = os.path.join(self.reslt_dir, self.taskname, 're_lasrgf.pth')

        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
            print('created ', self.task_dir)
        else:
            print('writing into', self.task_dir)


def main():
    args = arguments(opt)
    model = GoodModel(args)
    dataloader = make_loader(args)

if __name__ == '__main__':
    main()