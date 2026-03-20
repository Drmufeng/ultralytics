# sys
import os
import numpy as np
import json
# torch
import torch

# operation
from . import tools


class Feeder_kinetics(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=5,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3  #channel
        self.T = 300  #frame
        self.V = 17  #joint
        self.M = self.num_person_out  #person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    # def __getitem__(self, index):
    #     # output shape (C, T, V, M)
    #     # get data
    #     sample_name = self.sample_name[index]
    #     sample_path = os.path.join(self.data_path, sample_name)
    #     with open(sample_path, 'r') as f:
    #         video_info = json.load(f)
    #
    #     # fill data_numpy
    #     data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
    #     for frame_info in video_info['data']:
    #         frame_index = frame_info['frame_index']
    #         for m, skeleton_info in enumerate(frame_info["skeleton"]):
    #             if m >= self.num_person_in:
    #                 break
    #             pose = skeleton_info['pose']
    #             score = skeleton_info['score']
    #             data_numpy[0, frame_index, :, m] = pose[0::2]
    #             data_numpy[1, frame_index, :, m] = pose[1::2]
    #             data_numpy[2, frame_index, :, m] = score
    #
    #     # centralization
    #     data_numpy[0:2] = data_numpy[0:2] - 0.5
    #     data_numpy[0][data_numpy[2] == 0] = 0
    #     data_numpy[1][data_numpy[2] == 0] = 0
    #
    #     # get & check label index
    #     label = video_info['label_index']
    #     assert (self.label[index] == label)
    #
    #     # data augmentation
    #     if self.random_shift:
    #         data_numpy = tools.random_shift(data_numpy)
    #     if self.random_choose:
    #         data_numpy = tools.random_choose(data_numpy, self.window_size)
    #     elif self.window_size > 0:
    #         data_numpy = tools.auto_pading(data_numpy, self.window_size)
    #     if self.random_move:
    #         data_numpy = tools.random_move(data_numpy)
    #
    #     # sort by score
    #     sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    #     for t, s in enumerate(sort_index):
    #         data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
    #                                                                    0))
    #     data_numpy = data_numpy[:, :, :, 0:self.num_person_out]
    #
    #     # match poses between 2 frames
    #     if self.pose_matching:
    #         data_numpy = tools.openpose_match(data_numpy)
    #
    #     return data_numpy, label

    def __getitem__(self, index):
        """
        支持多人跟踪固定槽位版本。
        假设 JSON 中每帧 skeleton 已按固定顺序排列（来自 PoseTracker 槽位 0..M-1）
        不再做置信度排序或人数截断。
        """
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)

        with open(sample_path, 'r', encoding='utf-8') as f:
            video_info = json.load(f)

        # 初始化 (C, T, V, M)
        C, T, V, M = self.C, self.T, self.V, self.num_person_in
        data_numpy = np.zeros((C, T, V, M), dtype=np.float32)

        for frame_info in video_info.get('data', []):
            frame_index = frame_info.get('frame_index', 0)
            if not (0 <= frame_index < T):
                continue

            for m, skeleton_info in enumerate(frame_info.get("skeleton", [])):
                if m >= M:
                    break
                pose = np.asarray(skeleton_info.get('pose', []), dtype=np.float32)
                score = np.asarray(skeleton_info.get('score', []), dtype=np.float32)

                if len(pose) == 0 or len(score) == 0:
                    continue

                # 验证关键点数量
                if len(pose) != V * 2 or len(score) != V:
                    continue

                # 写入 (C, T, V, M)
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # 坐标中心化（如果原始是 [0,1]）
        data_numpy[0:2] -= 0.5
        mask = (data_numpy[2] == 0)
        data_numpy[0][mask] = 0
        data_numpy[1][mask] = 0
        data_numpy = np.nan_to_num(data_numpy, copy=False)

        # 取标签
        label = video_info['label_index']
        assert self.label[index] == label

        # 时序增强
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # 不再排序、不再截断，不做 pose_matching
        return data_numpy, label

        # 中心化（若坐标为 [0,1]）
        data_numpy[0:2] -= 0.5

        # 置信度为 0 的关键点置零坐标
        mask = (data_numpy[2] == 0)
        data_numpy[0][mask] = 0
        data_numpy[1][mask] = 0

        # 统一清洗，避免残留 NaN/Inf
        data_numpy = np.nan_to_num(data_numpy, copy=False)

        # 标签一致性检查
        label = video_info['label_index']
        assert (self.label[index] == label)

        # 数据增强
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # === 按每帧每人总置信度从高到低排序（安全不打乱轴顺序） ===
        # data_numpy[2] 的形状为 (T, V, M_in)；对 V 求和得到 (T, M_in)
        score_per_person = data_numpy[2].sum(axis=1)  # (T, M_in)
        order = np.argsort(-score_per_person, axis=1)  # (T, M_in) 降序

        for t in range(self.T):
            # 取出该帧 (C, V, M_in)
            tmp = data_numpy[:, t, :, :]
            # 沿人维 axis=2 重排
            data_numpy[:, t, :, :] = np.take(tmp, order[t], axis=2)

        # 只保留前 num_person_out 个人
        data_numpy = data_numpy[:, :, :, :self.num_person_out]

        # 帧间姿态匹配（可选）
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
