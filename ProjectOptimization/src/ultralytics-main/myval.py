from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('runs/pose/train49/weights/best.pt')
#     model.val(data='dataset/pose1/coco-pose1.yaml',
#               split='val',
#               batch=32,
#               # save_json=True,
#               project='runs/val',
#               name='exp',
#               )

if __name__ == '__main__':
    model = YOLO('yolo11n-pose.pt')
    model.val(data='dataset/pose/coco-pose.yaml',
              split='val',
              batch=16,
              # save_json=True,
              project='runs/val',
              name='exp',
              )