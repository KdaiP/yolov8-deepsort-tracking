from ultralytics import YOLO 
import math
import cv2 
import cvzone
import numpy as np
import tempfile
import os
import deep_sort.deep_sort.deep_sort as ds


#YoloV8官方模型标签数据，本次项目只使用了'person'
classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush']



#视频处理
def processVideo(inputPath,model):

    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7') #加载deepsort权重文件
    model=YOLO(model)#加载YOLO模型文件

    cap = cv2.VideoCapture(inputPath)#从inputPath读入视频
    fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取视频的大小
    output_viedo = cv2.VideoWriter()#初始化视频写入
    outputPath=tempfile.mkdtemp()#创建输出视频的临时文件夹的路径

    #输出格式为XVID格式的avi文件
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')#视频编码：h264,只能保存为avi格式,本地运行时不能在浏览器播放
    video_save_path = os.path.join(outputPath,"output.avi")#创建输出视频路径

    output_viedo.open(video_save_path , fourcc, fps, size, True)
    #对每一帧图片进行读取和处理
    while True:
        ret, img = cap.read()
        results=model(img,stream=True)
        detections=np.empty((0, 4))
        confarray = []
        if not(ret):
            break
        #读取推理的数据
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xywh[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将tensor类型转变为整型
                conf=math.ceil(box.conf[0]*100)/100#对conf取2位小数
                cls=int(box.cls[0])#获取物体类别标签
                #只检测和跟踪行人
                if cls==0:
                    currentArray=np.array([x1,y1,x2,y2])
                    confarray.append(conf)
                    detections=np.vstack((detections,currentArray))#按行堆叠数据

        #行人跟踪
        resultsTracker=tracker.update(detections, confarray, img)
        for result in resultsTracker:
            x1,y1,x2,y2,Id=result
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将浮点数转变为整型
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.putTextRect(img,f'{int(Id)}',(max(-10,x1),max(40,y1)),scale=1.3,thickness=2)
        output_viedo.write(img)#将处理后的图像写入视频
    output_viedo.release()#释放
    cap.release()#释放
    print(video_save_path)
    return video_save_path


if __name__ == '__main__':
    model = "yolov8n.pt"
    input_video_path = "test.mp4" #在这里填入视频文件路径
    processVideo(input_video_path, model)