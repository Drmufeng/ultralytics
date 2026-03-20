# 文件：tools/my_gendat.py
import os
import sys
import argparse
import pickle
import traceback
from pathlib import Path
from numpy.lib.format import open_memmap

# ========= 1) 路径与导入 =========
# 项目根目录动态解析：无须手动改盘符，迁移后仍可用。
ROOT = str(Path(__file__).resolve().parents[1])

try:
    if os.getcwd() != ROOT:
        os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
except Exception as e:
    print("⚠️ 切换或注入项目根目录失败：", repr(e))

try:
    from feeder.feeder_kinetics import Feeder_kinetics
    print("✅ 已导入 feeder.feeder_kinetics")
except Exception as e:
    print("❌ 导入 feeder 失败：", repr(e))
    traceback.print_exc()
    sys.exit(1)


# ========= 2) 进度条（可选） =========
toolbar_width = 30
def print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")


# ========= 3) 核心转换函数 =========
def gendata(
    data_path,
    label_path,
    data_out_path,
    label_out_path,
    num_person=5,     # M：每帧保留人数（in/out 同值，保证槽位一致）
    max_frame=300     # T：窗口长度
):
    """
    读取 Kinetics-style 的目录与标签，生成 ST-GCN 需要的 (N, 3, T, 17, M) numpy 数组与 label.pkl。
    注意：如果你已经在数据准备阶段用“跟踪+固定槽位”把人排好，
         那么这里的 Feeder_kinetics 不应再做“按分数重排/截人”，而是 in==out==M。
    """
    print(f"\n=== 开始转换 ===\n"
          f"data_path   : {data_path}\n"
          f"label_path  : {label_path}\n"
          f"data_out    : {data_out_path}\n"
          f"label_out   : {label_out_path}\n"
          f"num_person  : {num_person}\n"
          f"max_frame   : {max_frame}\n")

    # 1) 初始化 Feeder
    try:
        feeder = Feeder_kinetics(
            data_path=data_path,
            label_path=label_path,
            num_person_in=num_person,
            num_person_out=num_person,   # 关键：in/out 一致，避免再次裁人打乱槽位
            window_size=max_frame
        )
    except Exception as e:
        print("❌ 初始化 Feeder_kinetics 失败：", repr(e))
        traceback.print_exc()
        return

    sample_name = feeder.sample_name
    N = len(sample_name)
    print(f"样本数 N = {N}")
    if N == 0:
        print("⚠️ 没有样本可处理。请检查 data_path/label_path。")
        return

    # 2) 预分配 memmap：形状 (N, 3, T, 17, M)
    try:
        os.makedirs(os.path.dirname(data_out_path), exist_ok=True)
        fp = open_memmap(
            data_out_path,
            dtype='float32',
            mode='w+',
            shape=(N, 3, max_frame, 17, num_person)
        )
    except Exception as e:
        print("❌ 创建 memmap 失败：", repr(e))
        traceback.print_exc()
        return

    # 3) 逐样本写入
    sample_label = []
    for i, s in enumerate(sample_name):
        try:
            data, label = feeder[i]   # data 期待形状 (3, T_i, 17, M_i)

            # 形状检查与裁剪/填充
            if data.ndim != 4 or data.shape[0] != 3 or data.shape[2] != 17:
                raise ValueError(f"数据形状异常，期望 (3, T, 17, M)，实际 {data.shape}")

            Ti = min(data.shape[1], max_frame)
            Mi = min(data.shape[3], num_person)

            # 先清零该样本
            fp[i, ...] = 0

            # 将 data 写入 memmap
            fp[i, :, :Ti, :, :Mi] = data[:, :Ti, :, :Mi]

            sample_label.append(label)

            # 简单进度条
            if N >= 5:
                print_toolbar((i + 1) / N, annotation='写入进度 ')
            else:
                print(f"  √ 写入样本 {i+1}/{N}: {s}")
        except Exception as e:
            print(f"❌ 写入样本 {i} 失败（{s}）：", repr(e))
            traceback.print_exc()
            # 出错样本清零，继续处理其他样本
            fp[i, ...] = 0
            sample_label.append(-1)  # 或者跳过：看你是否要保持与 sample_name 对齐

    if N >= 5:
        end_toolbar()

    # 4) 保存 label.pkl
    try:
        with open(label_out_path, 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)
        print("✅ 已保存标签：", label_out_path)
    except Exception as e:
        print("❌ 保存 label.pkl 失败：", repr(e))
        traceback.print_exc()
        return

    print("🎉 转换完成！")


# ========= 4) 命令行入口 =========
def main():
    parser = argparse.ArgumentParser(description='Kinetics-skeleton → ST-GCN 数据转换')
    parser.add_argument('--data_path', default='runs/yoloJSON', help='输入数据根目录')
    parser.add_argument('--out_folder', default='runs/yoloJSON_out', help='输出目录（将写入 *_data.npy 与 *_label.pkl）')
    parser.add_argument('--max_frame', type=int, default=300, help='序列长度 T')
    parser.add_argument('--num_person', type=int, default=5, help='每帧保留人数 M（in/out 同值）')
    args = parser.parse_args()

    parts = ['train', 'val']

    # 仅创建一次输出目录
    os.makedirs(args.out_folder, exist_ok=True)

    for p in parts:
        data_path = f'{args.data_path}/{p}'
        label_path = f'{args.data_path}/{p}label.json'   # 修正：带下划线
        data_out_path = f'{args.out_folder}/{p}data.npy'
        label_out_path = f'{args.out_folder}/{p}label.pkl'

        # 基本检查
        if not os.path.isdir(data_path):
            print(f"⚠️ 数据目录不存在：{data_path}，跳过该分片")
            continue
        if not os.path.isfile(label_path):
            print(f"⚠️ 标签文件不存在：{label_path}，跳过该分片")
            continue

        try:
            gendata(
                data_path=data_path,
                label_path=label_path,
                data_out_path=data_out_path,
                label_out_path=label_out_path,
                num_person=args.num_person,
                max_frame=args.max_frame
            )
        except Exception as e:
            print(f"❌ 分片 {p} 处理异常：", repr(e))
            traceback.print_exc()

if __name__ == '__main__':
    main()
