import mne
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import re
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

# 数据路径设置
BASE_DIR = os.getcwd()
EDF_DIR = os.path.join(BASE_DIR, 'EDF', 'Experimental')
FMS_DIR = os.path.join(BASE_DIR, 'For Zenodo', 'Experimental Group')
EPOCHS_DIR = os.path.join(BASE_DIR, 'Epochs_Data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'Processed_Data')  # 新增预处理数据目录

# 创建输出目录
os.makedirs(EPOCHS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)  # 确保预处理目录存在

# 设置日志
LOG_FILE = os.path.join(EPOCHS_DIR, 'eeg_epoching.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 记录开始信息
logger.info("开始执行EEG分段处理")
logger.info(f"数据目录: {BASE_DIR}")
logger.info(f"输出目录: {EPOCHS_DIR}")

# 事件码定义 - 预期的事件码
EXPECTED_EVENT_IDS = {
    'cog_target': 16,
    'cog_distractor': 32,
    'baseline1_end': 37,
    'baseline2_end': 47,
    'baseline3_end': 57,
    'baseline4_end': 67,
    'tunnel_fms_report': 100,
    'coaster_fms_report': 102
}


# 提取受试者ID和对应的EDF文件
def get_subject_files():
    """获取所有实验组的EDF文件，并按受试者ID分组"""
    edf_files = glob.glob(os.path.join(EDF_DIR, 'E*.edf'))
    subject_files = {}

    for file in edf_files:
        # 文件名格式: E日期时间戳_受试者ID.edf
        file_name = os.path.basename(file)
        # 使用正则表达式提取受试者ID
        match = re.search(r'E\d+_(\d+)\.edf', file_name)
        if match:
            subject_id = int(match.group(1))
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file)

    return subject_files


# 读取FMS评分文件
def read_fms_scores(subject_id):
    """读取受试者的FMS评分文件"""
    subject_dir = os.path.join(FMS_DIR, str(subject_id))

    tunnel_fms_files = glob.glob(os.path.join(subject_dir, "*Gabor_FMS_backup.csv"))
    coaster_fms_files = glob.glob(os.path.join(subject_dir, "*Rollercoaster_FMS_backup.csv"))

    tunnel_fms = None
    coaster_fms = None

    if tunnel_fms_files:
        tunnel_fms = pd.read_csv(tunnel_fms_files[0], header=None, names=['Label', 'FMS'])

    if coaster_fms_files:
        coaster_fms = pd.read_csv(coaster_fms_files[0], header=None, names=['Label', 'FMS'])

    return tunnel_fms, coaster_fms


# 将FMS评分转换为类别
def fms_to_category(fms_score):
    """
    将FMS评分转换为类别:
    - 轻微 (0-5): 0
    - 中度 (6-10): 1
    - 严重 (11+): 2
    """
    if 0 <= fms_score <= 5:
        return 0  # 轻微
    elif 6 <= fms_score <= 10:
        return 1  # 中度
    else:
        return 2  # 严重


# 处理单个受试者的数据
def process_subject(subject_id, files):
    """处理单个受试者的数据文件，提取任务相关的EEG段"""
    logger.info(f"正在处理受试者 {subject_id} 的数据...")

    # 合并多个文件（如果有）
    if len(files) > 1:
        logger.info(f"受试者 {subject_id} 有多个EEG文件，将进行合并处理")
        raw_list = []
        for file in files:
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose='WARNING')
                raw_list.append(raw)
            except Exception as e:
                logger.error(f"读取文件 {file} 时出错: {e}")

        if raw_list:
            raw = mne.concatenate_raws(raw_list)
        else:
            logger.error(f"受试者 {subject_id} 的所有文件读取失败")
            return None, None, None, None
    else:
        try:
            raw = mne.io.read_raw_edf(files[0], preload=True, verbose='WARNING')
        except Exception as e:
            logger.error(f"读取文件 {files[0]} 时出错: {e}")
            return None, None, None, None

    # 打印通道信息
    logger.info(f"通道数量: {len(raw.ch_names)}")
    logger.info(f"通道列表: {raw.ch_names}")
    logger.info(f"采样率: {raw.info['sfreq']} Hz")

    # 应用带通滤波器 (0.1-40 Hz)
    raw.filter(l_freq=0.1, h_freq=40.0, verbose='WARNING')
    for ch_name in raw.ch_names:
        ch_data = raw.get_data(picks=[ch_name])
        ch_mean = np.mean(ch_data)
        raw.apply_function(lambda x: x - ch_mean, picks=[ch_name])

    # 保存预处理后的数据 - 使用FIF格式
    processed_file = os.path.join(PROCESSED_DIR, f'subject_{subject_id}_processed.fif')
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    raw.save(processed_file, overwrite=True)
    logger.info(f"预处理后的数据已保存至: {processed_file}")

    # 事件提取优化
    events = []
    event_id = {}
    event_id_map = {}  # 用于映射注释事件ID到预期事件ID

    try:
        try:
            # 先尝试直接找事件
            logger.info("尝试直接寻找事件...")
            events = mne.find_events(raw, stim_channel=None, verbose='WARNING')
            logger.info(f"通过find_events找到事件: {len(events)} 个")
        except Exception as e:
            logger.warning(f"未找到刺激通道，尝试从注释提取事件: {e}")
            events, event_id = mne.events_from_annotations(raw)
            logger.info(f"通过annotations找到事件: {len(events)} 个")
            logger.info(f"事件ID映射: {event_id}")

            # 创建从注释事件ID到预期事件ID的映射
            for annotation_desc, annotation_id in event_id.items():
                try:
                    # 尝试从注释描述中提取数字
                    match = re.search(r'\d+', annotation_desc)
                    if match:
                        event_code = int(match.group())
                        # 查找该事件码是否在预期的事件码中
                        for expected_name, expected_code in EXPECTED_EVENT_IDS.items():
                            if event_code == expected_code:
                                event_id_map[annotation_id] = expected_code
                                break
                    else:
                        # 如果注释描述中没有数字，尝试根据关键词匹配
                        if 'baseline1' in annotation_desc.lower() or 'baseline 1' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['baseline1_end']
                        elif 'baseline2' in annotation_desc.lower() or 'baseline 2' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['baseline2_end']
                        elif 'baseline3' in annotation_desc.lower() or 'baseline 3' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['baseline3_end']
                        elif 'baseline4' in annotation_desc.lower() or 'baseline 4' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['baseline4_end']
                        elif 'tunnel' in annotation_desc.lower() or 'gabor' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['tunnel_fms_report']
                        elif 'roller' in annotation_desc.lower() or 'coaster' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['coaster_fms_report']
                        elif 'target' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['cog_target']
                        elif 'distractor' in annotation_desc.lower():
                            event_id_map[annotation_id] = EXPECTED_EVENT_IDS['cog_distractor']
                except Exception as e:
                    logger.warning(f"处理注释 '{annotation_desc}' 时出错: {e}")

            # 打印映射结果
            logger.info(f"事件ID映射结果: {event_id_map}")

            # 将注释事件的ID替换为预期的ID
            if event_id_map:
                for i in range(len(events)):
                    if events[i, 2] in event_id_map:
                        events[i, 2] = event_id_map[events[i, 2]]
    except Exception as e:
        logger.error(f"事件提取完全失败: {e}，跳过事件部分")

    # 打印事件类型
    if len(events) > 0:
        unique_events = np.unique(events[:, 2])
        logger.info(f"找到的事件类型: {unique_events}")
    else:
        logger.warning("没有找到任何事件，无法进行分段处理")
        return None, None, None, None

    # 读取FMS评分
    tunnel_fms, coaster_fms = read_fms_scores(subject_id)

    if tunnel_fms is None and coaster_fms is None:
        logger.warning(f"未找到受试者 {subject_id} 的FMS评分文件")
        return None, None, None, None

    # 记录FMS评分
    if tunnel_fms is not None:
        logger.info(f"隧道任务FMS评分: {tunnel_fms['FMS'].tolist()}")
    if coaster_fms is not None:
        logger.info(f"过山车任务FMS评分: {coaster_fms['FMS'].tolist()}")

    # 初始化窗口类别计数器
    mild_count = 0  # 轻微症状 (0-5分)
    moderate_count = 0  # 中度症状 (6-10分)
    severe_count = 0  # 严重症状 (11+分)
    baseline_count = 0

    # 处理基线段
    logger.info("开始处理基线段数据...")
    baseline_epochs = []
    baseline_names = ['baseline1_end', 'baseline2_end', 'baseline3_end']
    baseline_codes = [EXPECTED_EVENT_IDS[name] for name in baseline_names]

    # 记录找到了哪些基线事件
    found_baselines = []
    for i, baseline_code in enumerate(baseline_codes):
        baseline_events = events[events[:, 2] == baseline_code]
        if len(baseline_events) > 0:
            found_baselines.append((i + 1, baseline_code, baseline_events))
            logger.info(f"找到基线{i + 1}结束事件: {len(baseline_events)} 个")
        else:
            logger.warning(f"未找到基线{i + 1}结束事件 (代码: {baseline_code})")

    # 设置窗口大小为20秒
    window_size = 20  # 窗口大小（秒）

    # 处理每个找到的基线事件
    for baseline_num, baseline_code, baseline_events in found_baselines:
        for event_idx, event in enumerate(baseline_events):
            # 获取事件时间点（秒）
            event_time = event[0] / raw.info['sfreq']

            # 计算段起始时间（事件前5分钟）
            segment_start = max(0, event_time - 5 * 60)
            segment_end = event_time

            # 确保段长度合理（至少要有一个完整的20秒窗口）
            if segment_end - segment_start < window_size:
                logger.warning(f"基线{baseline_num}事件 {event_idx} 前数据不足{window_size}秒，跳过")
                continue

            logger.info(
                f"提取基线{baseline_num}事件前的数据段: {segment_start:.1f}s - {segment_end:.1f}s (持续时间: {segment_end - segment_start:.1f}s)")

            # 提取基线段数据
            try:
                baseline_data = raw.copy().crop(segment_start, segment_end)
            except Exception as e:
                logger.error(f"提取基线{baseline_num}数据段时出错: {e}")
                continue

            # 将数据分割为20秒的epoch
            window_starts = np.arange(segment_start, segment_end - window_size + 0.1, window_size)
            baseline_count += len(window_starts)

            logger.info(f"基线{baseline_num}: 共创建 {len(window_starts)} 个{window_size}秒窗口")

            for w, window_start in enumerate(window_starts):
                window_end = window_start + window_size
                if window_end <= segment_end:  # 确保窗口不超出段范围
                    try:
                        window_data = raw.copy().crop(window_start, window_end)

                        # 添加到列表，FMS类别标记为3
                        baseline_epochs.append({
                            'data': window_data,
                            'task': f'baseline{baseline_num}',
                            'fms_score': 0,  # 基线数据没有FMS评分，设为0
                            'fms_category': 3,  # 基线数据类别标记为3
                            'window_index': w,
                            'overlap': 0.0
                        })
                    except Exception as e:
                        logger.error(f"创建基线{baseline_num}窗口 {w} 时出错: {e}")

    logger.info(f"基线处理完成，共创建 {len(baseline_epochs)} 个基线窗口")

    # 处理隧道任务段
    tunnel_epochs = []
    if tunnel_fms is not None:
        # 查找隧道任务基线结束事件和FMS报告事件
        baseline2_events = events[events[:, 2] == EXPECTED_EVENT_IDS['baseline2_end']]
        tunnel_fms_events = events[events[:, 2] == EXPECTED_EVENT_IDS['tunnel_fms_report']]

        logger.info(f"隧道任务基线结束事件数量: {len(baseline2_events)}")
        logger.info(f"隧道任务FMS报告事件数量: {len(tunnel_fms_events)}")

        # 如果找不到特定事件，尝试使用最接近的事件
        if len(baseline2_events) == 0 and len(events) > 0:
            logger.warning("未找到隧道任务基线结束事件，尝试使用事件序列创建分段")
            # 根据FMS评分数量和事件序列创建分段
            if len(tunnel_fms) > 0 and len(events) >= len(tunnel_fms):
                # 选择前n个事件，其中n为FMS评分数量
                all_events_sorted = events[np.argsort(events[:, 0])]
                # 为了简化处理，我们假设前len(tunnel_fms)个事件对应FMS评分
                segment_events = all_events_sorted[:len(tunnel_fms)]

                # 创建一个额外的起始事件
                if len(segment_events) > 0:
                    # 在第一个事件之前添加一个起始事件
                    first_event_time = segment_events[0, 0]
                    start_event = np.array([[max(0, first_event_time - int(300 * raw.info['sfreq'])), 0, 999]])
                    # 添加起始事件
                    baseline2_events = start_event
                    tunnel_fms_events = segment_events

        if len(tunnel_fms_events) > 0:
            # 对每个FMS报告事件进行处理
            for i, fms_event in enumerate(tunnel_fms_events):
                if i < len(tunnel_fms):
                    fms_score = tunnel_fms['FMS'].iloc[i]
                    fms_category = fms_to_category(fms_score)

                    # 确定段起始和结束：在事件前后各取30秒
                    event_time = fms_event[0] / raw.info['sfreq']  # 事件时间点（秒）
                    segment_start = max(0, event_time - 30)  # 事件前30秒
                    segment_end = min(raw.times[-1], event_time + 30)  # 事件后30秒

                    # 确保段长度合理
                    if segment_end - segment_start < 10:  # 至少需要10秒的数据
                        logger.warning(f"警告: 隧道任务FMS事件 {i} 附近的数据不足，跳过")
                        continue

                    # 提取数据段
                    segment_data = raw.copy().crop(segment_start, segment_end)
                    segment_duration = segment_end - segment_start

                    logger.info(f"隧道任务FMS事件 {i}: 持续时间={segment_duration:.2f}秒, FMS评分={fms_score}")

                    # 根据FMS类别确定窗口重叠率
                    if fms_category == 0:  # 轻微症状 - 无重叠
                        overlap = 0.0
                    elif fms_category == 1:  # 中度症状 - 60%重叠
                        overlap = 0.6
                    else:  # 严重症状 - 81.82%重叠
                        overlap = 0.8182

                    # 计算步长
                    step = window_size * (1 - overlap)

                    # 创建窗口
                    window_starts = np.arange(segment_start, segment_end - window_size + 0.1, step)

                    logger.info(f"隧道任务FMS事件 {i}: 类别={fms_category}, 重叠率={overlap:.2f}, 窗口数={len(window_starts)}")

                    # 更新窗口类别计数
                    if fms_category == 0:
                        mild_count += len(window_starts)
                    elif fms_category == 1:
                        moderate_count += len(window_starts)
                    else:
                        severe_count += len(window_starts)

                    for w, window_start in enumerate(window_starts):
                        window_end = window_start + window_size
                        if window_end <= segment_end:  # 确保窗口不超出段范围
                            window_data = raw.copy().crop(window_start, window_end)

                            # 添加到列表
                            tunnel_epochs.append({
                                'data': window_data,
                                'task': 'tunnel',
                                'fms_score': fms_score,
                                'fms_category': fms_category,
                                'window_index': w,
                                'overlap': overlap
                            })
    else:
        logger.warning(f"受试者 {subject_id} 未找到隧道任务FMS报告事件")

    # 处理过山车任务段 (使用相同的重叠率规则)
    coaster_epochs = []
    if coaster_fms is not None:
        # 查找过山车任务FMS报告事件
        coaster_fms_events = events[events[:, 2] == EXPECTED_EVENT_IDS['coaster_fms_report']]

        logger.info(f"过山车任务FMS报告事件数量: {len(coaster_fms_events)}")

        # 如果找不到特定事件，尝试使用事件序列创建分段
        if len(coaster_fms_events) == 0 and len(events) > 0:
            logger.warning("未找到过山车任务FMS报告事件，尝试使用事件序列创建分段")
            # 跳过前面已用于隧道任务的事件
            remaining_events = len(events) - len(tunnel_fms) if tunnel_fms is not None else len(events)

            if len(coaster_fms) > 0 and remaining_events >= len(coaster_fms):
                # 选择接下来的n个事件，其中n为FMS评分数量
                all_events_sorted = events[np.argsort(events[:, 0])]
                start_idx = len(tunnel_fms) if tunnel_fms is not None else 0
                end_idx = start_idx + len(coaster_fms)
                if end_idx <= len(all_events_sorted):
                    coaster_fms_events = all_events_sorted[start_idx:end_idx]

        if len(coaster_fms_events) > 0:
            # 对每个FMS报告事件进行处理
            for i, fms_event in enumerate(coaster_fms_events):
                if i < len(coaster_fms):
                    fms_score = coaster_fms['FMS'].iloc[i]
                    fms_category = fms_to_category(fms_score)

                    # 确定段起始和结束：在事件前后各取30秒
                    event_time = fms_event[0] / raw.info['sfreq']  # 事件时间点（秒）
                    segment_start = max(0, event_time - 30)  # 事件前30秒
                    segment_end = min(raw.times[-1], event_time + 30)  # 事件后30秒

                    # 确保段长度合理
                    if segment_end - segment_start < 10:  # 至少需要10秒的数据
                        logger.warning(f"警告: 过山车任务FMS事件 {i} 附近的数据不足，跳过")
                        continue

                    # 提取数据段
                    segment_data = raw.copy().crop(segment_start, segment_end)
                    segment_duration = segment_end - segment_start

                    logger.info(f"过山车任务FMS事件 {i}: 持续时间={segment_duration:.2f}秒, FMS评分={fms_score}")

                    # 根据FMS类别确定窗口重叠率 (使用与隧道任务相同的规则)
                    if fms_category == 0:  # 轻微症状 - 无重叠
                        overlap = 0.0
                    elif fms_category == 1:  # 中度症状 - 60%重叠
                        overlap = 0.6
                    else:  # 严重症状 - 81.82%重叠
                        overlap = 0.8182

                    # 计算步长
                    step = window_size * (1 - overlap)

                    # 创建窗口
                    window_starts = np.arange(segment_start, segment_end - window_size + 0.1, step)

                    logger.info(f"过山车任务FMS事件 {i}: 类别={fms_category}, 重叠率={overlap:.2f}, 窗口数={len(window_starts)}")

                    # 更新窗口类别计数
                    if fms_category == 0:
                        mild_count += len(window_starts)
                    elif fms_category == 1:
                        moderate_count += len(window_starts)
                    else:
                        severe_count += len(window_starts)

                    for w, window_start in enumerate(window_starts):
                        window_end = window_start + window_size
                        if window_end <= segment_end:  # 确保窗口不超出段范围
                            window_data = raw.copy().crop(window_start, window_end)

                            # 添加到列表
                            coaster_epochs.append({
                                'data': window_data,
                                'task': 'coaster',
                                'fms_score': fms_score,
                                'fms_category': fms_category,
                                'window_index': w,
                                'overlap': overlap
                            })
    else:
        logger.warning(f"受试者 {subject_id} 未找到过山车任务FMS报告事件")

    # 如果两种任务都没有找到有效分段，尝试基于FMS评分直接创建窗口
    if len(tunnel_epochs) == 0 and len(coaster_epochs) == 0:
        logger.warning("尝试基于FMS评分直接创建窗口，而不依赖事件")

        # 创建均匀分布的时间窗口
        total_duration = raw.times[-1]  # 总时长（秒）

        # 处理隧道任务
        if tunnel_fms is not None and len(tunnel_fms) > 0:
            # 估计隧道任务时长（假设占总时长的前半部分）
            tunnel_duration = total_duration / 2
            # 每个FMS评分对应的时长
            fms_duration = tunnel_duration / len(tunnel_fms)

            for i, fms_score in enumerate(tunnel_fms['FMS']):
                fms_category = fms_to_category(fms_score)
                # 计算该FMS评分对应的时间段
                segment_start = i * fms_duration
                segment_end = (i + 1) * fms_duration

                # 分成20秒的窗口
                n_windows = int((segment_end - segment_start) / window_size)

                logger.info(f"隧道任务备用段 {i}: 持续时间={fms_duration:.2f}秒, FMS评分={fms_score}, 窗口数={n_windows}")

                # 更新窗口类别计数
                if fms_category == 0:
                    mild_count += n_windows
                elif fms_category == 1:
                    moderate_count += n_windows
                else:
                    severe_count += n_windows

                for w in range(n_windows):
                    window_start = segment_start + w * window_size
                    window_end = window_start + window_size

                    # 确保不超出数据范围
                    if window_end <= total_duration:
                        window_data = raw.copy().crop(window_start, window_end)

                        # 添加到列表
                        tunnel_epochs.append({
                            'data': window_data,
                            'task': 'tunnel',
                            'fms_score': fms_score,
                            'fms_category': fms_category,
                            'window_index': w
                        })

        # 处理过山车任务
        if coaster_fms is not None and len(coaster_fms) > 0:
            # 估计过山车任务时长（假设占总时长的后半部分）
            coaster_start = total_duration / 2
            coaster_duration = total_duration / 2
            # 每个FMS评分对应的时长
            fms_duration = coaster_duration / len(coaster_fms)

            for i, fms_score in enumerate(coaster_fms['FMS']):
                fms_category = fms_to_category(fms_score)
                # 计算该FMS评分对应的时间段
                segment_start = coaster_start + i * fms_duration
                segment_end = coaster_start + (i + 1) * fms_duration

                # 分成20秒的窗口
                n_windows = int((segment_end - segment_start) / window_size)

                logger.info(f"过山车任务备用段 {i}: 持续时间={fms_duration:.2f}秒, FMS评分={fms_score}, 窗口数={n_windows}")

                # 更新窗口类别计数
                if fms_category == 0:
                    mild_count += n_windows
                elif fms_category == 1:
                    moderate_count += n_windows
                else:
                    severe_count += n_windows

                for w in range(n_windows):
                    window_start = segment_start + w * window_size
                    window_end = window_start + window_size

                    # 确保不超出数据范围
                    if window_end <= total_duration:
                        window_data = raw.copy().crop(window_start, window_end)

                        # 添加到列表
                        coaster_epochs.append({
                            'data': window_data,
                            'task': 'coaster',
                            'fms_score': fms_score,
                            'fms_category': fms_category,
                            'window_index': w
                        })

    # 合并两个任务的epochs
    all_epochs = tunnel_epochs + coaster_epochs + baseline_epochs

    if len(all_epochs) == 0:
        logger.error(f"受试者 {subject_id} 未能创建任何有效窗口，跳过保存")
        return 0, 0, 0, 0

    # 保存epochs
    subject_dir = os.path.join(EPOCHS_DIR, f"subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)

    # 记录epoch信息
    epoch_info = []

    for i, epoch in enumerate(all_epochs):
        # 保存原始数据
        epoch_file = os.path.join(subject_dir, f"epoch_{i}_{epoch['task']}_fms{epoch['fms_score']}.fif")
        epoch['data'].save(epoch_file, overwrite=True)

        # 记录信息
        epoch_info.append({
            'subject_id': subject_id,
            'epoch_index': i,
            'task': epoch['task'],
            'fms_score': epoch['fms_score'],
            'fms_category': epoch['fms_category'],
            'window_index': epoch['window_index'],
            'overlap': epoch.get('overlap', 0.0),  # 添加重叠率信息
            'file_path': epoch_file
        })

    # 保存epoch信息
    info_df = pd.DataFrame(epoch_info)
    info_file = os.path.join(subject_dir, 'epoch_info.csv')
    info_df.to_csv(info_file, index=False)

    # 显示各类别窗口数量
    logger.info(f"受试者 {subject_id} 的窗口分布情况:")
    logger.info(f"  轻微症状(0-5分)窗口: {mild_count} 个")
    logger.info(f"  中度症状(6-10分)窗口: {moderate_count} 个")
    logger.info(f"  严重症状(11+分)窗口: {severe_count} 个")
    logger.info(f"  基线窗口: {len(baseline_epochs)} 个")
    logger.info(f"  总计: {len(all_epochs)} 个窗口")
    logger.info(f"数据已保存至: {subject_dir}")

    # 返回窗口数量统计
    return len(all_epochs), mild_count, moderate_count, severe_count


# 主函数
def main():
    """主函数，处理所有受试者数据"""
    # 获取所有受试者文件
    subject_files = get_subject_files()

    logger.info(f"找到 {len(subject_files)} 名实验组受试者")

    # 处理每个受试者
    total_windows = 0
    total_mild = 0
    total_moderate = 0
    total_severe = 0

    for subject_id, files in subject_files.items():
        logger.info(f"\n开始处理受试者 {subject_id}...")
        result = process_subject(subject_id, files)
        if result and len(result) == 4:
            windows, mild, moderate, severe = result
            total_windows += windows
            total_mild += mild
            total_moderate += moderate
            total_severe += severe
        else:
            logger.warning(f"受试者 {subject_id} 处理失败")

    # 记录总体分布情况
    logger.info("\n" + "=" * 50)
    logger.info("全部受试者窗口分布情况:")
    logger.info(f"  轻微症状(0-5分)窗口: {total_mild} 个 ({total_mild / total_windows * 100:.1f}%)")
    logger.info(f"  中度症状(6-10分)窗口: {total_moderate} 个 ({total_moderate / total_windows * 100:.1f}%)")
    logger.info(f"  严重症状(11+分)窗口: {total_severe} 个 ({total_severe / total_windows * 100:.1f}%)")
    logger.info(f"  总计: {total_windows} 个窗口")
    logger.info("=" * 50)

    # 生成窗口类别分布图
    if total_windows > 0:
        plt.figure(figsize=(10, 6))
        categories = ['轻微 (0-5分)', '中度 (6-10分)', '严重 (11+分)']
        counts = [total_mild, total_moderate, total_severe]
        colors = ['green', 'orange', 'red']

        plt.bar(categories, counts, color=colors)
        plt.title('EEG窗口按症状严重程度分布', fontsize=16)
        plt.xlabel('症状严重程度', fontsize=14)
        plt.ylabel('窗口数量', fontsize=14)

        # 在柱状图上显示数量和百分比
        for i, count in enumerate(counts):
            percentage = count / total_windows * 100
            plt.text(i, count + 0.5, f'{count}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=12)

        # 保存图表
        chart_file = os.path.join(EPOCHS_DIR, 'window_distribution.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"窗口分布图已保存至: {chart_file}")

    logger.info(f"处理完成。总共处理了 {len(subject_files)} 名受试者，产生了 {total_windows} 个数据窗口。")


if __name__ == "__main__":
    main()