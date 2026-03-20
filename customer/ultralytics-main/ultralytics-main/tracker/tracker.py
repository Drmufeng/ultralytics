import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import cv2

class Track:
    def __init__(self, track_id, bbox, keypoints, feature=None):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.keypoints = keypoints  # 17个关键点 [x, y, confidence]
        self.feature = feature
        self.hits = 1
        self.time_since_update = 0
        self.state = 'Tentative'  # Tentative, Confirmed, Deleted
        self.trajectory = [keypoints]  # 存储关键点轨迹
        
    def update(self, bbox, keypoints, feature=None):
        self.bbox = bbox
        self.keypoints = keypoints
        self.feature = feature
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append(keypoints)
        
        # 保持轨迹长度
        if len(self.trajectory) > 30:  # 保持30帧的轨迹
            self.trajectory.pop(0)
            
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'
    
    def predict(self):
        # 简单的线性预测
        if len(self.trajectory) >= 2:
            last_kpts = np.array(self.trajectory[-1])
            second_last_kpts = np.array(self.trajectory[-2])
            velocity = last_kpts - second_last_kpts
            predicted_kpts = last_kpts + velocity
            return predicted_kpts.tolist()
        return self.keypoints
    
    def mark_missed(self):
        self.time_since_update += 1
        if self.time_since_update > 30:
            self.state = 'Deleted'

class PoseTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_extractor = self._build_feature_extractor()
    
    def _build_feature_extractor(self):
        """构建特征提取器用于Re-ID"""
        class SimpleFeatureExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(64, 128)
                )
            
            def forward(self, x):
                return F.normalize(self.conv(x), p=2, dim=1)
        
        return SimpleFeatureExtractor()
    
    def extract_features(self, img, bbox):
        """从检测框中提取特征用于Re-ID"""
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(128)
            
        crop = cv2.resize(crop, (64, 128))
        crop = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            feature = self.feature_extractor(crop).cpu().numpy().flatten()
        return feature
    
    def compute_iou(self, boxA, boxB):
        """计算IoU"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def compute_pose_similarity(self, kpts1, kpts2):
        """计算姿态相似度"""
        kpts1 = np.array(kpts1).reshape(-1, 3)
        kpts2 = np.array(kpts2).reshape(-1, 3)
        
        # 只考虑可见的关键点
        valid_mask = (kpts1[:, 2] > 0.5) & (kpts2[:, 2] > 0.5)
        if not np.any(valid_mask):
            return 0.0
        
        # 计算欧几里得距离
        distances = np.linalg.norm(kpts1[valid_mask, :2] - kpts2[valid_mask, :2], axis=1)
        similarity = np.exp(-np.mean(distances) / 50.0)  # 归一化
        return similarity
    
    def compute_cost_matrix(self, img, detections):
        """计算代价矩阵"""
        tracks = [t for t in self.tracks.values() if t.state != 'Deleted']
        
        if len(tracks) == 0 or len(detections) == 0:
            return np.array([]), tracks, detections
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # IoU相似度
                iou = self.compute_iou(track.bbox, detection['bbox'])
                
                # 姿态相似度
                pose_sim = self.compute_pose_similarity(track.keypoints, detection['keypoints'])
                
                # 特征相似度
                det_feature = self.extract_features(img, detection['bbox'])
                feature_sim = np.dot(track.feature, det_feature) if track.feature is not None else 0
                
                # 综合相似度
                total_sim = 0.4 * iou + 0.4 * pose_sim + 0.2 * feature_sim
                cost_matrix[i, j] = 1 - total_sim  # 转换为代价
        
        return cost_matrix, tracks, detections
    
    def update(self, img, detections):
        """更新跟踪器"""
        # 预测所有轨迹的下一个位置
        for track in self.tracks.values():
            if track.state != 'Deleted':
                track.predict()
        
        # 计算匹配代价
        cost_matrix, tracks, detections = self.compute_cost_matrix(img, detections)
        
        matched_indices = []
        if cost_matrix.size > 0:
            # 使用匈牙利算法进行匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:  # 阈值
                    matched_indices.append((row, col))
        
        # 更新匹配的轨迹
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for row, col in matched_indices:
            track = tracks[row]
            detection = detections[col]
            
            # 提取特征
            feature = self.extract_features(img, detection['bbox'])
            
            # 更新轨迹
            track.update(detection['bbox'], detection['keypoints'], feature)
            
            unmatched_detections.remove(col)
            unmatched_tracks.remove(row)
        
        # 处理未匹配的检测 - 创建新轨迹
        for idx in unmatched_detections:
            detection = detections[idx]
            feature = self.extract_features(img, detection['bbox'])
            
            new_track = Track(
                track_id=self.next_id,
                bbox=detection['bbox'],
                keypoints=detection['keypoints'],
                feature=feature
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # 处理未匹配的轨迹
        for idx in unmatched_tracks:
            tracks[idx].mark_missed()
        
        # 删除长时间未更新的轨迹
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.state == 'Deleted':
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracks[track_id]
        
        # 返回确认的轨迹
        confirmed_tracks = []
        for track in self.tracks.values():
            if track.state == 'Confirmed':
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'keypoints': track.keypoints,
                    'trajectory': track.trajectory
                })
        
        return confirmed_tracks

    def train_feature_extractor(self, train_loader, epochs=50, lr=0.001):
        """训练特征提取器用于Re-ID"""
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=lr)
        criterion = torch.nn.TripletMarginLoss(margin=0.2)
        
        self.feature_extractor.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                anchor, positive, negative = batch
                
                anchor_features = self.feature_extractor(anchor)
                positive_features = self.feature_extractor(positive)
                negative_features = self.feature_extractor(negative)
                
                loss = criterion(anchor_features, positive_features, negative_features)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        self.feature_extractor.eval()
        print("特征提取器训练完成！")