from ultralytics import YOLO

# if __name__ == '__main__':
#     # 加载模型
#     model = YOLO("ultralytics/cfg/models/11/yolopose+MAN-Faster+ContextGuidedDown1/yolo11-pose-LSDECD.yaml")  # 从头开始构建新模型
#     print(model)
#
#     # Use the model
#     results = model.train(data="dataset/pose1/coco-pose1.yaml",  epochs=500, batch=32, device='0')

if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11-pose-LSDECD.yaml",task="pose")  # 从头开始构建新模型
    # print(model)

    # Use the model
    results = model.train(data="dataset/pose/coco-pose.yaml",  epochs=300, batch=16, device='0')