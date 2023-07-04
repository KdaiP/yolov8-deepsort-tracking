# yolov8-deepsort-tracking
opencv+yolov8+deepsort的行人检测与跟踪。当然，也可以识别车辆等其他类别。

2023/7/4更新：
加入了一个基于Gradio的WebUI界面

![示例图片](https://github.com/KdaiP/yolov8-deepsort-tracking/blob/main/demo.png)

## 安装（不管用不用WebUI都要装）
本项目需要pytorch，建议手动在[pytorch官网](https://pytorch.org/get-started/locally/)根据自己的平台和CUDA环境安装对应的版本。

安装完pytorch后，可通过以下命令来安装依赖：

```shell
$ pip install -r requirements.txt
```

如果你不知道pytorch是什么，也不知道CUDA是什么，只是在赶课程项目的进度的话，也可以直接使用上面这条命令，会自动安装pytorch。

## 配置(非WebUI)

在app.py中修改以下代码，将输入视频路径换成你要处理的视频的路径：

```python
input_video_path = "test.mp4"
```

模型默认使用Ultralytics官方的YOLOv8n模型：

```python
model = "yolov8n.pt"
```

第一次使用会自动从官网下载模型，如果网速过慢，可以在[ultralytics的官方文档](https://docs.ultralytics.com/tasks/detect/)下载模型，然后将模型文件拷贝到程序所在目录下。

## 运行(非WebUI)

运行app.py
运行完成后，终端会显示输出视频所在的路径。

## WebUI界面的配置和运行

**请先确保已经安装完成上面的依赖**

安装Gradio库：

```shell
$ pip install gradio
```

运行gradio_webui.py，如果控制台出现以下消息代表成功运行：
```shell
Running on local URL:  http://127.0.0.1:6006
To create a public link, set `share=True` in `launch()`
```

浏览器打开该URL即可使用WebUI界面

![WebUI](https://github.com/KdaiP/yolov8-deepsort-tracking/blob/main/webui.png)

