# # muti_video_generator.py
#
# from ultralytics import YOLO
# import torch
# import numpy as np
# import cv2, time, os
# import os
# import json
#
#
# def InputReader(path):
#     cap = cv2.VideoCapture("{}".format(path))  # 视频流读取
#     rate = cap.get(5)  # 帧速率
#     FrameNumber = cap.get(7)  # 视频文件的帧数
#     duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
#     return cap, rate, FrameNumber, duration
#
#
# # def single_video_output(img, model):
# #     result = model(img, imgsz=320, conf=0.5)[0]
# #     # custom.visualize(img, result)
# #     a = result.keypoints.conf
# #     b = result.keypoints.xyn
# #
# #     # 如果预测结果为空，则补0；a补n*1，b补n*2，n为关节点数目
# #     if a == None:
# #         a = torch.zeros(17).unsqueeze(0)
# #         b = torch.zeros(34).unsqueeze(0).unsqueeze(0)  # 需要  扩维度
# #     # 输出 关键点
# #     np1 = b[0].cpu().numpy()
# #     np2 = np.around(np1, 3).flatten()
# #
# #     # 输出 置信度
# #     conf1 = a.cpu().numpy()
# #     conf2 = np.around(conf1, 3)[0]
# #
# #     point = [round(i, 3) for i in np2]
# #     conf = [round(i, 3) for i in conf2]
# #     return point, conf
# def single_video_output(img, model):
#     result = model(img, imgsz=320, conf=0.5)[0]
#
#     # 确保 result.keypoints 非空
#     if result.keypoints is None or len(result.keypoints.xyn) == 0:
#         # 如果没有关键点检测结果，返回空数据
#         return [], []
#
#     # custom.visualize(img, result)  # 可视化关键点
#     a = result.keypoints.conf  # 关键点置信度
#     b = result.keypoints.xyn  # 关键点坐标
#
#     # 如果预测结果为空，则补0；a补n*1，b补n*2，n为关节点数目
#     if a is None or b is None or len(b) == 0:
#         a = torch.zeros(17).unsqueeze(0)
#         b = torch.zeros(34).unsqueeze(0).unsqueeze(0)  # 需要扩维度
#
#     # 输出关键点
#     np1 = b[0].cpu().numpy() if len(b) > 0 else np.zeros(34)
#     np2 = np.around(np1, 3).flatten()
#
#     # 输出置信度
#     conf1 = a.cpu().numpy()
#     conf2 = np.around(conf1, 3)[0] if conf1 is not None else np.zeros(17)
#
#     point = [round(i, 3) for i in np2]
#     conf = [round(i, 3) for i in conf2]
#
#     return point, conf
#
#
# class single_video_json_output():
#
#     def __init__(self):
#         self.model = YOLO('yolo11n-pose.pt')  # 此处修改 model路径
#         self.capture = 'test.mp4'
#         self.class_name = 'none'
#
#     def _inference(self, name):
#         cap, rate, FrameNumber, duration = InputReader(self.capture)
#         labels = ['BaseballPitch', 'Basketball', 'Bowling', 'CricketBowling']
#
#         # json存储
#         frame_index = 0
#         jsdata = []
#
#         while cap.isOpened():
#             rec, img = cap.read()
#             if rec == False:
#                 break
#             frame_index += 1
#
#             # pose_data = {"pose": '', "score": ''}
#             # frame_data = {"frame_index": 0, "skeleton": []}
#
#             point, conf = single_video_output(img, self.model)
#             point = [round(float(x), 3) for x in point]  # 数据保留小数点位数
#             conf = [round(float(x), 3) for x in conf]
#
#             # 字典内部存储 此处需要修改 因为有多个人的视频
#             pose_data = {"pose": point, "score": conf}
#             frame_data = {"frame_index": frame_index, "skeleton": [pose_data]}  # 注意加个括号
#             if sum(point) == 0:
#                 frame_data = {"frame_index": frame_index, "skeleton": []}
#             jsdata.append(frame_data)
#
#             cv2.imshow('1', img)
#             del img
#             # 按q结束
#             if cv2.waitKey(1) == ord('q'):
#                 break
#
#         output_data = {
#             "data": jsdata,
#             "label": self.class_name,
#             "label_index": labels.index(self.class_name)
#         }
#
#         # 拼接文件路径
#         file_path = 'runs/yoloJSON/' + self.class_name + '/' + name.split('.')[0] + '.json'
#
#         # 确保目录存在
#         directory = os.path.dirname(file_path)
#         os.makedirs(directory, exist_ok=True)
#         # with open(file_path, "w") as json_file:
#         #     json.dump(output_data, json_file, indent=None, separators=(',', ':'))
#         with open(file_path, "w") as json_file:
#             json.dump(output_data, json_file, indent=4, separators=(',', ': '), ensure_ascii=False)
#
#
# # 运行 所有类别
# if __name__ == '__main__':
#
#     all_path = 'dataset/GCNData/'  # 视频文件夹总路径
#     classes = os.listdir(all_path)
#
#     for single_class in classes:
#         PATH = all_path + single_class + '/'
#         a = single_video_json_output()
#
#         videos = os.listdir(PATH)
#
#         for video_name in videos:
#             a.class_name = single_class
#             a.capture = PATH + video_name
#             a._inference(video_name)
import json

import numpy as np
import torch

from ultralytics import YOLO

import cv2
import os

def InputReader(path):
    cap = cv2.VideoCapture("{}".format(path))  # 视频流读取
    rate = cap.get(5)  # 帧速率
    FrameNumber = cap.get(7)  # 视频文件的帧数
    duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
    return cap, rate, FrameNumber, duration


def single_video_output(img, model):
    result = model(img, imgsz=320, conf=0.5)[0]

    # 确保 result.keypoints 非空
    if result.keypoints is None or len(result.keypoints.xyn) == 0:
        return [], []

    a = result.keypoints.conf  # 关键点置信度
    b = result.keypoints.xyn  # 关键点坐标

    # 如果预测结果为空，则补0；a补n*1，b补n*2，n为关节点数目
    if a is None or b is None or len(b) == 0:
        a = torch.zeros(17).unsqueeze(0)
        b = torch.zeros(34).unsqueeze(0).unsqueeze(0)  # 需要扩维度

    points = []
    confs = []

    # 遍历每个人的关键点数据
    for i in range(len(b)):
        np1 = b[i].cpu().numpy() if len(b) > 0 else np.zeros(34)
        np2 = np.around(np1, 3).flatten()

        conf1 = a[i].cpu().numpy() if a is not None else np.zeros(17)
        conf2 = np.around(conf1, 3)

        point = [round(float(i), 3) for i in np2]  # 转换为 Python 的 float 类型
        conf = [round(float(i), 3) for i in conf2]  # 转换为 Python 的 float 类型

        points.append(point)
        confs.append(conf)

    return points, confs


class single_video_json_output():

    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')  # 修改模型路径
        self.capture = 'test.mp4'
        self.class_name = 'none'

    def _inference(self, name):
        cap, rate, FrameNumber, duration = InputReader(self.capture)
        labels = ['BaseballPitch', 'Basketball', 'Bowling', 'CricketBowling']

        # json存储
        frame_index = 0
        jsdata = []

        while cap.isOpened():
            rec, img = cap.read()
            if rec == False:
                break
            frame_index += 1

            points, confs = single_video_output(img, self.model)

            frame_data = {"frame_index": frame_index, "skeleton": []}

            # 如果没有人检测到，跳过当前帧
            if sum([sum(point) for point in points]) == 0:
                jsdata.append(frame_data)
                continue

            # 每个人的关键点和置信度
            for point, conf in zip(points, confs):
                pose_data = {"pose": point, "score": conf}
                frame_data["skeleton"].append(pose_data)

            jsdata.append(frame_data)

            cv2.imshow('1', img)
            del img

            if cv2.waitKey(1) == ord('q'):
                break

        output_data = {
            "data": jsdata,
            "label": self.class_name,
            "label_index": labels.index(self.class_name)
        }

        file_path = 'runs/JSON/' + self.class_name + '/' + name.split('.')[0] + '.json'

        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        # 解决 numpy 类型问题，将数据转换为标准的 float 类型
        with open(file_path, "w") as json_file:
            json.dump(output_data, json_file, indent=4, separators=(',', ': '), ensure_ascii=False)


# 运行 所有类别
if __name__ == '__main__':

    all_path = 'dataset/GCNData/'  # 视频文件夹总路径
    classes = os.listdir(all_path)

    for single_class in classes:
        PATH = all_path + single_class + '/'
        a = single_video_json_output()

        videos = os.listdir(PATH)

        for video_name in videos:
            a.class_name = single_class
            a.capture = PATH + video_name
            a._inference(video_name)
