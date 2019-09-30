import torch
import torchvision

class block(torch.nn.Module):
    def __init__(self, in_f, out_f, addpass=False):
        super(block, self).__init__()
        self.addpass = addpass
        self.tunnel = torch.nn.Sequential(
                torch.nn.Conv2d(in_f, out_f, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(out_f, out_f, kernel_size=3),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(True),
                # torch.nn.Conv2d(out_f, out_f, kernel_size=1, bias=False),
                # torch.nn.BatchNorm2d(out_f),
                # torch.nn.ReLU(True),
            )

    def forward(self, x):
        out = self.tunnel(x)
        return out


class Yolo_res101_large(torch.nn.Module):
    def __init__(self, pretrained=True, small=True):
        super(Yolo_res101_large, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resbone = torch.nn.Sequential(*(list(resnet.children())[:-2])) # --> 2048 14 14

        self.tunnel1 = block(2048, 256)     # 12 12
        self.tunnel2 = block(256, 256)      # 10 10
        self.tunnel3 = block(256, 128)      # 8  8
        self.yolo2 = torch.nn.Sequential(        # --> 26 7 7 
                torch.nn.Conv2d(128, 26, kernel_size=2, bias=False),
                torch.nn.BatchNorm2d(26),
            )
        # self.tunnel = torch.nn.Sequential(
        #         torch.nn.Conv2d(2048,256, kernel_size=3, stride=1, bias=False),  # 12
        #         torch.nn.BatchNorm2d(256),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False),   # 10
        #         torch.nn.BatchNorm2d(256),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, bias=False),   # 8
        #         torch.nn.BatchNorm2d(128),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(128, 26, kernel_size=2, stride=1, bias=False),   # 7 
        #         torch.nn.BatchNorm2d(26),
        #         torch.nn.ReLU(True),
        #     )
        
    def forward(self,z):
        out = self.resbone(z)

        x = self.tunnel1(out)
        x = self.tunnel2(x)
        x = self.tunnel3(x)
        x = self.yolo2(x)
        x = torch.sigmoid(x)
        x = x.permute(0,2,3,1)
        return x



## resnet[:-1] will be N, 2048, 8, 8, resnet[:-2] will be N, 2048, 14,14




class Yolo_res101(torch.nn.Module):
    def __init__(self, pretrained=True, small=True):
        super(Yolo_res101, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resbone = torch.nn.Sequential(*(list(resnet.children())[:-2])) # --> 2048 14 14
        # self.tunnel = torch.nn.Sequential(
        #         torch.nn.Conv2d(2048,256, kernel_size=3, stride=1, bias=False),  # 12
        #         torch.nn.BatchNorm2d(256),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False),   # 10
        #         torch.nn.BatchNorm2d(256),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, bias=False),   # 8
        #         torch.nn.BatchNorm2d(128),
        #         torch.nn.ReLU(True),
        #         torch.nn.Conv2d(128, 26, kernel_size=2, stride=1, bias=False),   # 7 
        #         torch.nn.BatchNorm2d(26),
        #         torch.nn.ReLU(True),
        #     )
        self.tunnel = torch.nn.Sequential(
                torch.nn.Conv2d(2048,256, kernel_size=3, stride=1, bias=False),  # 12
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, bias=False),   # 10
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
            )

        self.yolo = torch.nn.Sequential(
                torch.nn.Linear(6400, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 1274) #7*7*26
            )

    def forward(self,z):
        out = self.resbone(z)

        x = self.tunnel(out)        # N 64 10 10
        x = x.view(x.size(0), -1)   # N 6400
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1, 7, 7, 26)
        return x



## resnet[:-1] will be N, 2048, 8, 8, resnet[:-2] will be N, 2048, 14,14


