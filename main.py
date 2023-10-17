from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
import deep_sort.deep_sort.deep_sort as ds

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。

    :param img: 输入图像。
    :param text: 要绘制的文本。
    :param origin: 文本的左上角坐标。
    :param font: 字体类型。
    :param font_scale: 字体大小。
    :param text_color: 文本的颜色。
    :param bg_color: 背景的颜色。
    :param thickness: 文本的线条厚度。
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # 减去5以留出一些边距
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)  # 从左上角的位置减去5来留出一些边距
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

# 视频处理
def processVideo(inputPath: str) -> Path:
    """处理视频，检测并跟踪行人。

    :param inputPath: 视频文件路径
    :return: 输出视频的路径
    """
    # 读取视频文件
    cap = cv2.VideoCapture(inputPath)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )  # 获取视频的大小
    output_video = cv2.VideoWriter()  # 初始化视频写入

    # 输出格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_save_path = Path(outputPath) / "output.avi"  # 创建输出视频路径

    output_video.open(video_save_path.as_posix(), fourcc, fps, size, isColor=True)

    # 对每一帧图片进行读取和处理
    while True:
        success, frame = cap.read()
        if not (success):
            break

        # 获取每一帧的目标检测推理结果
        results = model(frame, stream=True)

        detections = np.empty((0, 4)) # 存放bounding box结果
        confarray = [] # 存放每个检测结果的置信度

        # 读取目标检测推理结果
        # 参考： https://docs.ultralytics.com/modes/predict/#working-with-results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xywh[0])  # 提取矩形框左上和右下的点，并将tensor类型转为整型
                conf = round(float(box.conf[0]), 2)  # 对conf四舍五入到2位小数
                cls = int(box.cls[0])  # 获取物体类别标签

                if cls == detect_class:
                    detections = np.vstack((detections,np.array([x1,y1,x2,y2])))
                    confarray.append(conf)

        # 使用deepsort进行跟踪
        resultsTracker = tracker.update(detections, confarray, frame)
        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 绘制bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)  # 将处理后的图像写入视频
    output_video.release()  # 释放
    cap.release()  # 释放
    print(f'output dir is: {video_save_path}')
    return video_save_path


if __name__ == "__main__":
    # 在这里填入视频文件路径
    ######
    input_video_path = "test.mp4"
    ######

    # 输出文件夹，默认为系统的临时文件夹路径
    outputPath = tempfile.mkdtemp()  # 创建临时文件夹用于存储输出视频

    # 加载yoloV8模型权重
    model = YOLO("yolov8n.pt")

    # 需要跟踪的物体类别，model.names返回模型所支持的所有物体类别
    # yoloV8官方模型的第一个类别为'person'
    detect_class = 0
    print(f"detecting {model.names[detect_class]}")

    # 加载deepsort模型权重
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    processVideo(input_video_path)
