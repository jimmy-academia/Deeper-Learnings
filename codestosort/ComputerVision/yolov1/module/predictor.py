import os
import sys
sys.path.append('module')
sys.path.append('utils')

from module.yolovgg import Yolov1_vgg16bn 
from module.yoloresnet import Yolo_res101
# from module.better_resnet import Yolo_betres101

from utils.evaluation import evaluate
from utils.visualization import visualize
from utils.prediction import _decoder

import torch
import torchvision
from tqdm import tqdm
from PIL import Image

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

class Predictor():
    def __init__(self, args):
        self.args = args
        self.modeltype = args.modeltype

        print('building', self.modeltype)
        if args.modeltype =='vgg':
            self.model = Yolov1_vgg16bn(pretrained=False)
        elif args.modeltype =='res':
            self.model = Yolo_res101(pretrained=False)
        elif args.modeltype == 'res50':
            self.model = Yolo_res101(pretrained=False, small=False)

        elif args.modeltype =='betres':
            self.model = Yolo_betres101(pretrained=True)
        elif args.modeltype == 'betres50':
            self.model = Yolo_betres101(pretrained=True, small=False)
        
        else:
            input('wrong modeltype, exiting')
            sys.exit(0)

        self.model_checkpoint = os.path.join('ckpt', self.args.taskname) 
        self.predict_dir = os.path.join(self.model_checkpoint, 'predict')
        self.graph_dir = os.path.join(self.model_checkpoint, 'graph')
        self._init_check()
        
        self.valid_dir = 'hw2_train_val/val1500'
        self.img_dir = os.path.join(self.valid_dir, 'images')
        self.answer_dir = os.path.join(self.valid_dir, 'labelTxt_hbb')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.to(self.device)

    def _init_check(self):
        exitmsg = 'checkpoint not found, wrong taskname or not trained'
        self._handle_dir_exist(self.model_checkpoint, exitmsg=exitmsg, warn=None)
        self._handle_dir_exist(self.predict_dir)
        self._handle_dir_exist(self.graph_dir)

    @staticmethod
    def _handle_dir_exist(_dir, exitmsg=None, warn='WARNING: %s exists, will write over'):
        '''
        usage:
        (dir) --> warn overwrite/create
        (dir, existmsg) --> warn overwrite/ exit if not found
        (dir, warn=None) --> no warn/create
        '''

        if os.path.exists(_dir):
            if warn is not None:
                warn = warn%str(_dir)
                input(warn)
        else:
            if exitmsg is not None:
                print(exitmsg)
                sys.exit(0)
            else:
                os.mkdir(_dir)

    def _loadmodel(self, file):
        path = os.path.join(self.model_checkpoint, file)
        self.model.load_state_dict(torch.load(path))
        print('load model from %s success!'%path)

    def _createprediction(self, pred_dir):
        print('creating predictions in', pred_dir)
        self._handle_dir_exist(pred_dir)
        if input('overwrite?') =='n':
            return 0
        for path in tqdm(os.listdir(self.img_dir)):
            imgpath = os.path.join(self.img_dir, path)
            filename = path.split('/')[-1].split('.')[0]
            filepath = os.path.join(pred_dir, filename+'.txt')
            self._singleprediction(imgpath, filepath)

    def _singleprediction(self, imgpath, anspath):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(448),
            torchvision.transforms.ToTensor(),
        ])
        img = Image.open(imgpath)
        img = trans(img)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        pred = self.model(img)
        pred = pred.squeeze(0).cpu()
        boxes,class_index,probs =  _decoder(pred)
        with open(anspath, 'w') as f:
            for i,box in enumerate(boxes):
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                # input([xmin, xmax, ymin, ymax])
                probability = probs[i]
                classname = classnames[int(class_index[i])]
                formats = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, classname, probability]
                line = str('{} {} {} {} {} {} {} {} {} {}\n'.format(*formats))
                f.write(line)
            # result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])

    def evaluate_(self, name):
        file = name + '.pth'
        print('evaluating for', self.model_checkpoint, file)
        self._loadmodel(file)
        pred_dir = os.path.join(self.predict_dir, name)
        self._createprediction(pred_dir)
        print('hi')
        map_ = evaluate(pred_dir, self.answer_dir)
        return map_

    def evaluate_list(self, _list = [1,10,20]):
        pass

    def draw_abox(self, name):
        file = name + '.pth'
        print('drawing for', self.model_checkpoint, file)
        print('(overwriting bbox)')
        self._loadmodel(file)
        
        for filename in ['0076', '0086', '0907']:
            print(filename)
            imgpath = os.path.join(self.img_dir, '{}.jpg'.format(filename))
            anspath = os.path.join(self.graph_dir, 'a_{}.txt'.format(filename))
            graphpath = os.path.join(self.graph_dir, 'a_{}.jpg'.format(filename))
            self._singleprediction(imgpath, anspath)
            visualize(imgpath, anspath, graphpath)

    def draw_bbox(self):
        print('drawing visualization for', self.model_checkpoint)
        for epoch in [1,10,20]:
            print('epoch', epoch)
            file = 'yolo_{}.pth'.format(epoch)
            self._loadmodel(file)
            for filename in ['0076', '0086', '0907']:
                print(filename)
                imgpath = os.path.join(self.img_dir, '{}.jpg'.format(filename))
                anspath = os.path.join(self.graph_dir, 'e{}_{}.txt'.format(epoch, filename))
                graphpath = os.path.join(self.graph_dir, 'e{}_{}.jpg'.format(epoch, filename))
                self._singleprediction(imgpath, anspath)
                visualize(imgpath, anspath, graphpath)

def main():
    predictor = Predictor('ckpt/vgg')
    # predictor.evaluate_best()
    predictor.draw_bbox()
    # predictor = Predictor('ckpt/res', modeltype='res')


if __name__ == '__main__':
    main()

