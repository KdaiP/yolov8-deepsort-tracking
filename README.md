# yolov8-deepsort-tracking
opencv+yolov8+deepsort的行人检测与跟踪。当然，也可以识别车辆等其他类别。

![示例图片](https://github.com/KdaiP/yolov8-deepsort-tracking/blob/main/demo.png)

## 安装
本项目需要pytorch，建议手动在[官网](https://pytorch.org/get-started/locally/)根据自己的平台和CUDA环境安装对应的版本。

安装完pytorch后，可通过以下命令来安装依赖：

```shell
$ pip install -r requirements.txt
```

如果你不知道pytorch是什么，也不知道CUDA是什么，只是在赶课程项目的进度的话，也可以直接使用上面这条命令，会自动安装pytorch。

## 配置

在程序中修改以下代码，将输入视频路径换成你要处理的视频的路径：

```python
input_video_path = "test03.mp4"
```

模型默认使用Ultralytics官方的YOLOv8n模型：

```python
model = "yolov8n.pt"
```

第一次使用会自动从官网下载模型，如果网速过慢，可以在[官网](https://docs.ultralytics.com/tasks/detect/)下载模型，然后将模型文件拷贝到程序所在目录下。

## 运行

```shell
python app.py
```

视频推理和编码完成后，会在终端输出输出视频所在的路径。