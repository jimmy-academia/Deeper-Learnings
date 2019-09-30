import gc
import os
import sys
sys.path.append('module')

from module.yolovgg import Yolov1_vgg16bn 
from module.yoloresnet import Yolo_res101, Yolo_res101_large 

from module.evaluation import evaluate
from module.visualization import visualize
from module.prediction import _decoder

from module.datafunc import make_dataloader
from module.yololoss import myloss

import torch
import torchvision

from tqdm import tqdm
from PIL import Image

# classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

classnames = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane']

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args):
        self.args = args
        self.modeltype = args.modeltype
        print('building', self.modeltype)
        if args.modeltype =='vgg':
            self.model = Yolov1_vgg16bn(pretrained=True)
            # for param in list(self.model.children())[0]:
                # param.requires_grad = False
        elif args.modeltype =='res':
            self.model = Yolo_res101(pretrained=True)
            # for param in list(self.model.children())[0]:
                # param.requires_grad = False
        elif args.modeltype == 'res50':
            self.model = Yolo_res101_large(pretrained=True)

        else:
            input('wrong modeltype, exiting')
            sys.exit(0)

        self.trainloader, self.validloader = make_dataloader(
        # self.trainloader = make_dataloader(
            self.args.data_dir, batch_size=self.args.batch_size)
        self.criterion = myloss()

        # self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr= 0.001, weight_decay=5e-4)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 0.001, weight_decay=5e-4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= 0.001, momentum=0.9, weight_decay=5e-4)
        
        # self._adjust_optimizer(1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.to(self.device)
        print(self.model)

        self.outdir = os.path.join('ckpt', self.args.taskname)
        self.best_path = os.path.join(self.outdir, 'best.pth')
        self.yolo_path = os.path.join(self.outdir, 'yolo.pth')

        self.model_checkpoint = os.path.join('ckpt', self.args.taskname) 
        self.predict_dir = os.path.join(self.model_checkpoint, 'predict')
        self.graph_dir = os.path.join(self.model_checkpoint, 'graph')
        
        self.valid_dir = 'hw2_train_val/val1500'
        self.img_dir = os.path.join(self.valid_dir, 'images')
        self.answer_dir = os.path.join(self.valid_dir, 'labelTxt_hbb')

        self._handle_exist()
        self._init_check()

    def _handle_exist(self):
        if os.path.exists(self.outdir):
            print('listdir:')
            os.listdir(self.outdir)
            if input('task exists, resume?') =='':
                print('resuming...')
                path = os.path.join(self.outdir, 'yolo.pth')
                print('loading model from', path)
                self.model.load_state_dict(torch.load(path))
                filenames = os.listdir(self.outdir)
                filenames = [i for i in filenames if 'yolo_' in i]
                last_epoch = max([int(i.split('.')[0].split('_')[-1]) for i in filenames])
                if self.args.res_epoch is not None:
                    self.start_epoch = self.args.res_epoch
                else:
                    self.start_epoch = last_epoch + 1
                
                print('set start_epoch to', self.start_epoch)
                self.best_path = os.path.join(self.outdir, 'best2.pth')
                print('set best path to', self.best_path)
                input('start?')
            else:
                print('overwriting')
                self.start_epoch = 0

        else:
            os.mkdir(self.outdir)
            self.start_epoch = 0

    def _init_check(self):
        exitmsg = 'checkpoint not found, wrong taskname or not trained'
        self._handle_dir_exist(self.model_checkpoint, exitmsg=exitmsg, warn=None)
        self._handle_dir_exist(self.predict_dir)
        self._handle_dir_exist(self.graph_dir)

    def _adjust_optimizer(self, new_lr):
        # print('new lr is ', new_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        
        self.best_map = -1
        self.best_loss = 9999999


        for epoch in range(self.start_epoch, self.args.epochs):


            self.model.train()
            if epoch in [0,1,3,6]:
                self.setlr = 0.0001  # for vgg
                self.newsetlr = 0.001
                self.train_epoch(epoch, warm=True)
            elif epoch==10:
                self.setlr = 0.001
                self.newsetlr = 0.002
                self.train_epoch(epoch, warm=True)
            elif epoch==20:
                self.setlr = 0.002
                self.newsetlr = 0.001
                self.train_epoch(epoch, warm=True)
            elif epoch==50:
                self.setlr = 0.001
                self.newsetlr = 0.0005
                self.train_epoch(epoch, warm=True)
            elif epoch==65:
                self.setlr = 0.0005
                self.newsetlr = 0.0001
                self.train_epoch(epoch, warm=True)
            elif epoch==80:
                self.setlr = 0.0001
                self.newsetlr = 0.00001
                self.train_epoch(epoch, warm=True)

            else:
                self.train_epoch(epoch)

            torch.cuda.empty_cache()                
            gc.collect()
            self.valid_epoch()
            torch.cuda.empty_cache()                
            gc.collect()

            map1_ = self.evaluate_(thresh=0.01)
            map2_ = self.evaluate_(thresh=0.05)
            map3_ = self.evaluate_(thresh=0.1)
            print([map1_, map2_, map3_])
            map_ = max([map1_, map2_, map3_])

            if map_ > self.best_map:
                self.best_map = map_
                save_path = os.path.join(self.outdir, 'bestmap.pth')
                torch.save(self.model.state_dict(), save_path)
                print('>> map:', map_, 'is current best')
            else:
                print('>> map:', map_,)

            
            torch.save(self.model.state_dict(), self.yolo_path)
            self.draw_abox()

            if epoch %10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.outdir, 'yolo_{}.pth'.format(epoch)))

    def valid_epoch(self):
        v_loss = 0.0
        pbar = tqdm(self.validloader, ncols=100)
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            pred = self.model(images)
            loss = self.criterion(pred,targets)
            v_loss += loss.data.item()
            pbar.set_postfix(loss = loss.item())
            
            del images, targets, loss, pred

        v_loss /= len(self.validloader)
        if self.best_loss > v_loss:
            self.best_loss = v_loss
            print('get best test loss %.5f' % self.best_loss)
        else:
            print('get test loss %.5f'%v_loss)

        del pbar

    def train_epoch(self,epoch, warm=False, newlr=1):
        pbar = tqdm(self.trainloader, ncols=100)
        pbar.set_description('%d '%epoch)
        length = len(self.trainloader)
        j = 0
        avg_loss=0
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            pred = self.model(images)
            loss = self.criterion(pred, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            j+=1
            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), avg=avg_loss/j)
            if warm:
                new_lr = self.setlr + (self.newsetlr-self.setlr)*j/length
                # new_lr = 1e-4 + (1e-3-1e-4)*j/length
                self._adjust_optimizer(new_lr)
                pbar.set_postfix(loss=loss.item(), avg=avg_loss/j, lr=new_lr)

            del pred, images, targets, loss
            # break
        del pbar


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
                # input(warn)
        else:
            if exitmsg is not None:
                print(exitmsg)
                sys.exit(0)
            else:
                os.mkdir(_dir)

    def _createprediction(self, pred_dir, thresh=0.01):
        # print('creating predictions in', pred_dir)
        self._handle_dir_exist(pred_dir)
        # if input('overwrite?') =='n':
            # return 0
        for path in os.listdir(self.img_dir):
        # for path in tqdm(os.listdir(self.img_dir), ncols=100):
            imgpath = os.path.join(self.img_dir, path)
            filename = path.split('/')[-1].split('.')[0]
            filepath = os.path.join(pred_dir, filename+'.txt')
            self._singleprediction(imgpath, filepath, thresh)

    def _singleprediction(self, imgpath, anspath, thresh=0.01):
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
        boxes,class_index,probs =  _decoder(pred, thresh)
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

    def evaluate_(self, thresh=0.01):
        # print('making predictions')
        self._createprediction(self.predict_dir, thresh)
        # print('evaluating')
        map_ = evaluate(self.predict_dir, self.answer_dir)
        return map_

    def draw_abox(self):
        for filename in ['0076', '0086', '0907', '0100', '0932']:
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

        
        

if __name__ == '__main__':
    class config():
        def __init__(self):
            self.epochs = 50
            self.lr = 0.0001
            self.taskname = 'vgg'

    from models import Yolov1_vgg16bn 
    from datafunc import make_dataloader
    from loss import loss

    args = config()
    trainer = Trainer(args)
    trainer.train()