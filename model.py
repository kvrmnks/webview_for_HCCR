import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),

            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
        )
        self.linear = nn.Linear(12800, 3755, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class MyResize(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img):
        raw_width, raw_height = img.size
        ratio = min(self.height / raw_height, self.width / raw_width)
        twidth, theight = (min(int(ratio * raw_width), self.width), min(int(ratio * raw_height), self.height))
        img = img.resize((twidth, theight), Image.ANTIALIAS)
        # 拼接图片，补足边框 居中
        ret = Image.new('L', (self.width, self.height), 255)
        ret.paste(img, (int((self.width - twidth) / 2), int((self.height - theight) / 2)))
        return ret


net = Net()
net.load_state_dict(torch.load('./123')['model'])
net.eval()


def eval(arr):
    height, width = arr.shape[0], arr.shape[1]
    new_arr = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            new_arr[i][j] = 255 - arr[i][j][3]
    from torchvision import transforms, utils

    img = Image.fromarray(new_arr)

    # img 为获得的图片

    img = MyResize(64, 64)(img)

    img_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)
    in_data = transforms.ToTensor()(img)
    in_left_right_data = transforms.ToTensor()(img_left_right)
    in_top_bottom_data = transforms.ToTensor()(img_top_bottom)

    with torch.no_grad():
        y_hat = net(in_data.view(1, 1, 64, -1))
        y_hat_left_right = net(in_left_right_data.view(1, 1, 64, -1))
        y_hat_top_bottom = net(in_top_bottom_data.view(1, 1, 64, -1))

    return [y_hat.argmax().item(), y_hat_left_right.argmax().item(), y_hat_top_bottom.argmax().item()]