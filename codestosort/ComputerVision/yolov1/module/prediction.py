import torch


# def _decoder(pred):
#     for i in range(7):
#         for j in range(7):
#             cell = pred[i][j]
            
def encoder(boxes, labels): ## length is ratio

    target = torch.zeros((7,7,26))
    cell_size = 1./7
    width_height = boxes[:,2:] - boxes[:,:2]
    center_x_ys = (boxes[:,2:]+ boxes[:,:2])/2

    for it, center_x_y in enumerate(center_x_ys):
        cell_i_j = (center_x_y/cell_size).ceil() - 1
        i = int(cell_i_j[1])
        j = int(cell_i_j[0])
        target[i, j, 4] = 1
        target[i, j, 9] = 1
        target[i, j, int(labels[it]+9)] = 1
        tl_x_y = cell_i_j * cell_size
        delta_x_y = (center_x_y - tl_x_y) / cell_size
        target[i, j, 2:4] = width_height[it]
        target[i, j, :2] = delta_x_y
        target[i, j, 7:9] = width_height[it]
        target[i, j, 5:7] = delta_x_y

    return target


def _decoder(pred, thresh=0.01):
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 448/7
    img_size = 448.
    pred = pred.data
    pred = pred.squeeze(0) #7x7x26
    contain1 = pred[:,:,4].unsqueeze(2)  # 7,7,1
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2) # 7,7,2

    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)

    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(7):
        for j in range(7):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]

                    # box is cx cy w h in (cell ratio; img ratio)
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell (in pixel)
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image (in pixel)
                    box[2:] = box[2:]*img_size
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,y1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    large = torch.ones(box_xy.shape) * 512
                    small = torch.zeros(box_xy.shape)
                    box_xy = torch.where(box_xy>512, large, box_xy)
                    box_xy = torch.where(box_xy<0, small, box_xy)
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > thresh:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.unsqueeze(0))
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]
    
def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.shape == torch.Size([]):
            i = order.item()
        else:
            i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


classnames = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane']

def reader(label_path):

    boxes = []
    labels = []

    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        obj_ = line.strip().split()
        xmin = float(obj_[0])
        ymin = float(obj_[1])
        xmax = float(obj_[4])
        ymax = float(obj_[5])
        obj_class = classnames.index(obj_[8]) + 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj_class)        ### +1???

    return boxes, labels

'''
from module.prediction import encoder, reader, _decoder
from module.augment import data_augmentation, processimg
import cv2
import torch
from PIL import Image

img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
boxes, labels = reader(label_path)
boxes = torch.Tensor(boxes)
labels = torch.Tensor(labels)


img, boxes, labels = data_augmentation(img, boxes, labels)

h, w, __ = img.shape
boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

img = processimg(img)
target = encoder(boxes, labels)

boxes,cls_indexs,probs = _decoder(target)
'''