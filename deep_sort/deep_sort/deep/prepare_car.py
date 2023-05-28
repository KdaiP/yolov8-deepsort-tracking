# -*- coding:utf8 -*-

import os
from PIL import Image
from shutil import copyfile, copytree, rmtree, move

PATH_DATASET = './car-dataset' # 需要处理的文件夹
PATH_NEW_DATASET = './car-reid-dataset' # 处理后的文件夹
PATH_ALL_IMAGES = PATH_NEW_DATASET + '/all_images'
PATH_TRAIN = PATH_NEW_DATASET + '/train'
PATH_TEST = PATH_NEW_DATASET + '/test'

# 定义创建目录函数
def mymkdir(path):
    path = path.strip() # 去除首位空格
    path = path.rstrip("\\") # 去除尾部 \ 符号
    isExists = os.path.exists(path) # 判断路径是否存在
    if not isExists:
        os.makedirs(path) # 如果不存在则创建目录
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        self.path = PATH_DATASET # 表示需要命名处理的文件夹

    # 修改图像尺寸
    def resize(self):
        for aroot, dirs, files in os.walk(self.path):
            # aroot是self.path目录下的所有子目录（含self.path）,dir是self.path下所有的文件夹的列表.
            filelist = files  # 注意此处仅是该路径下的其中一个列表
            # print('list', list)

            # filelist = os.listdir(self.path) #获取文件路径
            total_num = len(filelist)  # 获取文件长度（个数）

            for item in filelist:
                if item.endswith('.jpg'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                    src = os.path.join(os.path.abspath(aroot), item)

                    # 修改图片尺寸到128宽*256高
                    im = Image.open(src)
                    out = im.resize((128, 256), Image.ANTIALIAS)  # resize image with high-quality
                    out.save(src)  # 原路径保存

    def rename(self):

        for aroot, dirs, files in os.walk(self.path):
            # aroot是self.path目录下的所有子目录（含self.path）,dir是self.path下所有的文件夹的列表.
            filelist = files  # 注意此处仅是该路径下的其中一个列表
            # print('list', list)

            # filelist = os.listdir(self.path) #获取文件路径
            total_num = len(filelist)  # 获取文件长度（个数）

            i = 1  # 表示文件的命名是从1开始的
            for item in filelist:
                if item.endswith('.jpg'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                    src = os.path.join(os.path.abspath(aroot), item)

                    # 根据图片名创建图片目录
                    dirname = str(item.split('_')[0])
                    # 为相同车辆创建目录
                    #new_dir = os.path.join(self.path, '..', 'bbox_all', dirname)
                    new_dir = os.path.join(PATH_ALL_IMAGES, dirname)
                    if not os.path.isdir(new_dir):
                        mymkdir(new_dir)

                    # 获得new_dir中的图片数
                    num_pic = len(os.listdir(new_dir))

                    dst = os.path.join(os.path.abspath(new_dir),
                                       dirname + 'C1T0001F' + str(num_pic + 1) + '.jpg')
                    # 处理后的格式也为jpg格式的，当然这里可以改成png格式    C1T0001F见mars.py filenames 相机ID，跟踪指数
                    # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                    try:
                        copyfile(src, dst) #os.rename(src, dst)
                        print ('converting %s to %s ...' % (src, dst))
                        i = i + 1
                    except:
                        continue
            print ('total %d to rename & converted %d jpgs' % (total_num, i))
            
    def split(self):
        #---------------------------------------
        #train_test
        images_path = PATH_ALL_IMAGES
        train_save_path = PATH_TRAIN
        test_save_path = PATH_TEST
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(test_save_path)
        
        for _, dirs, _ in os.walk(images_path, topdown=True):
            for i, dir in enumerate(dirs):
                for root, _, files in os.walk(images_path + '/' + dir, topdown=True):
                    for j, file in enumerate(files):
                        if(j==0): # test dataset；每个车辆的第一幅图片
                            print("序号：%s  文件夹： %s  图片：%s 归为测试集" % (i + 1, root, file))
                            src_path = root + '/' + file
                            dst_dir = test_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
                        else:
                            src_path = root + '/' + file
                            dst_dir = train_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
        rmtree(PATH_ALL_IMAGES)
        
if __name__ == '__main__':
    demo = BatchRename()
    demo.resize()
    demo.rename()
    demo.split()


