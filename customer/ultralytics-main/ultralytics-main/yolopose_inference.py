import cv2
import torch
from ultralytics import YOLO

# 加载模型
model = YOLO(r"runs/pose/train/weights/best.pt")

# 读图
image_path = r"dataset/COCO/000000000036.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"图片读取失败: {image_path}")

# 推理（关闭日志更干净）
results = model(image, verbose=False)

# 如果用GPU，做一次同步，避免时序问题
if torch.cuda.is_available():
    torch.cuda.synchronize()

result = results[0]

# 取关键点
kpts = []
if result.keypoints is not None and result.keypoints.data is not None:
    kpts = result.keypoints.data.cpu().numpy()

# 画关键点
for person_kpts in kpts:
    for x, y, conf in person_kpts:
        if conf > 0.3:
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

# 画框（可选）
if result.boxes is not None and result.boxes.xyxy is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 保存结果（先保存，确保即使窗口不弹也有输出）
cv2.imwrite("output_image.jpg", image)
print("结果已保存为 output_image.jpg")

# 显示结果（更稳）
cv2.namedWindow("YOLOPose Inference", cv2.WINDOW_NORMAL)
cv2.imshow("YOLOPose Inference", image)
cv2.waitKey(1)
cv2.waitKey(0)
cv2.destroyAllWindows()