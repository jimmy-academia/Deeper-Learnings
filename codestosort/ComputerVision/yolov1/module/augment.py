import cv2
import random
import torch
import numpy as np
from PIL import Image

def data_augmentation(img, boxes, labels):
    img, boxes = RandomFlipHV(img, boxes, mode='h')
    img, boxes = RandomFlipHV(img, boxes, mode='v')
    img,boxes = RandomScaleHV(img,boxes, mode='h')
    img,boxes = RandomScaleHV(img,boxes, mode='v')
    img = RandomBlur(img)
    img = RandomHSV(img, mode='h')
    img = RandomHSV(img, mode='s')
    img = RandomHSV(img, mode='v')

    img,boxes,labels = RandomShift(img,boxes,labels)
    img,boxes,labels = RandomCrop(img,boxes,labels)
    return img, boxes, labels

#boxes N x (xmin ymin xmax ymax p) Torch tensor
#img is bgr

def RandomFlipHV(img, boxes, p=0.5, mode='h'):
    if random.random() > p:
        return img, boxes
    else:
        if mode == 'h':
            img_flip = np.fliplr(img).copy()
            __, w, __ = img.shape
            boxes[:,0], boxes[:,2] = w - boxes[:,2], w - boxes[:,0]
        elif mode == 'v':
            img_flip = np.flipud(img).copy()
            h, __, __ = img.shape
            boxes[:,1], boxes[:,3] = h - boxes[:,3], h - boxes[:,1]
        return img_flip, boxes

def RandomScaleHV(img, boxes, p=0.5, mode='h'):
    if random.random() > p:
        return img, boxes
    else:
        h, w ,__ = img.shape

        if mode == 'h':
            nw = random.randint(int(0.8*w), int(1.2*w))
            scale = nw/w
            img = cv2.resize(img, (nw, h))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
        elif mode == 'v':
            nh = random.randint(int(0.8*h), int(1.2*h))
            scale = nh/h
            img = cv2.resize(img, (w, nh) )
            scale_tensor = torch.FloatTensor([[1, scale, 1, scale]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        return img, boxes

def RandomBlur(img, p=0.5):
    if random.random() < p:
        img = cv2.blur(img, (5,5))
    return img

def RandomHSV(img, p=0.5, mode='h'): #hue, saturation, brightness
    if random.random() < p:
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        if mode == 'h':
            h = np.clip(h*adjust, 0, 255).astype(hsv.dtype)
        elif mode =='s':
            s = np.clip(s*adjust, 0, 255).astype(hsv.dtype)
        elif mode =='s':
            v = np.clip(v*adjust, 0, 255).astype(hsv.dtype)

        hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def RandomShift(img, boxes, labels, p=0.5):
    if random.random() < p:
        h, w, c = img.shape
        final_image = np.zeros((h,w,c), dtype=img.dtype)
        final_image[:,:,:] = (104,117,123) ## bgr
        x_delta = int(random.uniform(- 0.2 * w, 0.2 * w))
        y_delta = int(random.uniform(- 0.2 * h, 0.2 * h))

        final_image[max(0, y_delta) : min(h, h+y_delta), max(0, x_delta) : min(w, w+x_delta), : ] \
            = img[max(0,-y_delta) : min(h, h-y_delta), max(0, -x_delta) : min(w, w-x_delta), : ] 
        
        center = (boxes[:,2:]+boxes[:,:2])/2
        shift_x_y = torch.FloatTensor([[x_delta, y_delta]]).expand_as(center)

        x_y_min = boxes[:,:2] + shift_x_y
        x_y_max = boxes[:,2:] + shift_x_y

        mask = (x_y_min[:, 0] > 0) & (x_y_min[:, 1] > 0) & (x_y_max[:, 0] < w) & (x_y_max[:, 1] < h)
        mask = mask.view(-1,1)
        # mask1 = (x_y_min[:, 0] > 0) & (x_y_min[:, 1] > 0)
        # mask2 = (x_y_max[:, 0] < w) & (x_y_max[:, 1] < h)
        # mask = (mask1 & mask2).view(-1,1)
        new_boxes = boxes[mask.expand_as(boxes)].view(-1,4)
        if len(new_boxes) == 0:
            return img,boxes,labels

        box_shift = torch.FloatTensor([[x_delta, y_delta, x_delta, y_delta]]).expand_as(new_boxes)
        new_boxes += box_shift
        new_labels = labels[mask.view(-1)]
        return final_image, new_boxes, new_labels

    return img,boxes,labels

def RandomCrop(img, boxes, labels, p=0.5):
    if random.random() < p:
        center = (boxes[:,2:]+boxes[:,:2])/2
        h, w, c = img.shape
        nw = int(random.uniform(0.6*w, 0.95*w))
        nh = int(random.uniform(0.6*h, 0.95*h))
        x_min = int(random.uniform(0, w-nw))
        y_min = int(random.uniform(0, h-nh))

        x_y_min = boxes[:,:2].clone()
        x_y_max = boxes[:,2:].clone()

        mask1 = (x_y_min[:, 0] > x_min) & (x_y_min[:, 1] > y_min)
        mask2 = (x_y_max[:, 0] < (x_min+nw)) & (x_y_max[:, 1] < (y_min+nh)) # 1 is true, 0 is false, * is and
        mask = (mask1 & mask2).view(-1,1)

        new_boxes = boxes[mask.expand_as(boxes)].view(-1,4)
        if(len(new_boxes)==0):
            return img,boxes,labels 
        
        box_shift = torch.FloatTensor([[x_min,y_min,x_min,y_min]]).expand_as(new_boxes)
        new_boxes -= box_shift
        new_labels = labels[mask.view(-1)]
        final_image = img[y_min:y_min+nh, x_min:x_min+nw, :]
        return final_image, new_boxes, new_labels
    return img,boxes,labels


def processimg(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # mean = np.array(((123,117,104)), dtype=np.uint8) # rgb
    # img -= mean
    img = Image.fromarray(img)
    return img

if __name__ == '__main__':
    ## check augmentations:

    from prediction import encoder, reader, _decoder
    from visualization import visualcheck, draw
    import torchvision

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor(),
        ])

    def regroup(image, label):
        torchvision.utils.save_image(image, 'tmp.jpg')
        cv2image = cv2.imread('tmp.jpg')
        boxes, class_index, probs =  _decoder(label, 0.1)
        result = []
        for box, clsid, prob in zip(boxes, class_index, probs):
            xmin = int(float(box[0]))            
            ymin = int(float(box[1]))            
            xmax = int(float(box[2]))            
            ymax = int(float(box[3]))            
            result.append([(xmin,ymin), (xmax,ymax), clsid.item(), prob.item()])
        return cv2image, result

    def too(img, boxes):
        h, w, __ = img.shape
        # boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes).to(self.device)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        img = processimg(img)
        target = encoder(boxes, labels)
        return img, target

    visualcheck(path='tmp/orig.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)
    img, target = too(img, boxes)


    img = trans(img)
    
    img, result = regroup(img, target)
    draw(img, result, 'tmp/0.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)
    img, boxes = RandomFlipHV(img, boxes, mode='h')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/1.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img, boxes = RandomFlipHV(img, boxes, mode='v')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/2.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img,boxes = RandomScaleHV(img,boxes, mode='h')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/3.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img,boxes = RandomScaleHV(img,boxes, mode='v')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/4.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img = RandomBlur(img)
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/5.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img = RandomHSV(img, mode='h')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/6.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img = RandomHSV(img, mode='s')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/7.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img = RandomHSV(img, mode='v')
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/8.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img,boxes,labels = RandomShift(img,boxes,labels)
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/9.jpg')

    img = cv2.imread('hw2_train_val/train15000/images/00000.jpg')
    label_path = 'hw2_train_val/train15000/labelTxt_hbb/00000.txt'
    boxes, labels = reader(label_path)
    boxes = torch.Tensor(boxes)
    labels = torch.Tensor(labels)

    img,boxes,labels = RandomCrop(img,boxes,labels)
    img, target = too(img, boxes)
    img = trans(img)
    img, result = regroup(img, target)
    draw(img, result, 'tmp/10.jpg')




