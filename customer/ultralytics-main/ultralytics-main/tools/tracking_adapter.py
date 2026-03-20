# tools/tracking_adapter.py
import numpy as np
from collections import defaultdict

class SlotAssigner:
    """把 track_id 固定映射到 [0..M-1] 的槽位"""
    def __init__(self, max_slots):
        self.max_slots = max_slots
        self.id2slot = {}
        self.slot2id = [None] * max_slots

    def assign(self, tid: int):
        if tid in self.id2slot:
            return self.id2slot[tid]
        # 还有空位
        for i in range(self.max_slots):
            if self.slot2id[i] is None:
                self.slot2id[i] = tid
                self.id2slot[tid] = i
                return i
        # 满了：不再分配（也可自定义替换策略）
        return None

    def remove(self, tid: int):
        if tid in self.id2slot:
            s = self.id2slot.pop(tid)
            if 0 <= s < self.max_slots and self.slot2id[s] == tid:
                self.slot2id[s] = None


def fill_missing_forward(data):
    """
    data: (3, T, V, M)  置信度为0的帧用前一帧填充，平滑缺失
    """
    C, T, V, M = data.shape
    for m in range(M):
        for v in range(V):
            for t in range(1, T):
                if data[2, t, v, m] == 0 and data[2, t-1, v, m] > 0:
                    data[:, t, v, m] = data[:, t-1, v, m]
    return data


def build_tensor_from_tracker(frames, detections_per_frame, pose_tracker, M=5, V=17, T_max=300):
    """
    参数：
      frames: [np.ndarray(H,W,3)]，视频帧列表
      detections_per_frame: 长度与 frames 相同的列表，每项是该帧的 detection 字典列表：
           detection = {
              'bbox': [x1,y1,x2,y2],
              'keypoints': [[x,y,score], ... 17 个]
           }
      pose_tracker: 你的 PoseTracker 实例
      M: 每帧最多人数
      V: 关键点数（17）
      T_max: 截断的帧数

    返回：
      data: (3, T, V, M)  的 float32
    """
    T = min(len(frames), T_max)
    data = np.zeros((3, T, V, M), dtype=np.float32)

    slot_assigner = SlotAssigner(M)

    for t in range(T):
        img = frames[t]
        dets = detections_per_frame[t]  # 列表[ {bbox, keypoints}, ... ]

        # 更新跟踪器
        confirmed = pose_tracker.update(img, dets)  # [{'track_id', 'bbox', 'keypoints', 'trajectory'}...]

        # 把 confirmed 结果写入固定槽位
        for item in confirmed:
            tid = int(item['track_id'])
            slot = slot_assigner.assign(tid)
            if slot is None:   # 槽满，跳过（或定制替换策略）
                continue

            kpts = np.array(item['keypoints'], dtype=np.float32).reshape(V, 3)
            data[0, t, :, slot] = kpts[:, 0]  # x
            data[1, t, :, slot] = kpts[:, 1]  # y
            data[2, t, :, slot] = kpts[:, 2]  # score

        # 可选：对长时间未更新的轨迹进行释放，保持槽健康（取决于 PoseTracker 的删除策略）
        # 这里依赖你 PoseTracker.mark_missed/Deleted 的逻辑，必要时在 adapter 层同步调用 slot_assigner.remove(tid)

    # 缺帧前向填充，避免突然清零
    data = fill_missing_forward(data)
    return data
