import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

'''
特征提取器：
提取对应bounding box中的特征, 得到一个固定维度的embedding作为该bounding box的代表，
供计算相似度时使用。

模型训练是按照传统ReID的方法进行，使用Extractor类的时候输入为一个list的图片，得到图片对应的特征。
'''

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            # RGB图片数据范围是[0-255]，需要先经过ToTensor除以255归一化到[0,1]之后，
            # 再通过Normalize计算(x - mean)/std后，将数据归一化到[-1,1]。
            transforms.ToTensor(),
            # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]是从imagenet训练集中算出来的
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

# __call__()是一个非常特殊的实例方法。该方法的功能类似于在类中重载 () 运算符，
# 使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

