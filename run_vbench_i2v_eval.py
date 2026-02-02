#!/usr/bin/env python
"""
VBench I2V 8-GPU 并行评估脚本

用法:
    torchrun --nproc_per_node=8 run_vbench_i2v_eval.py --video_dir /path/to/videos

参数:
    --video_dir: 视频所在目录 (mp4文件直接放在这个目录下)
    --output_dir: 输出目录 (默认为 video_dir/vbench_output)
"""

import argparse
import os
import json
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='VBench I2V 8-GPU 并行评估')
    parser.add_argument('--video_dir', type=str,
                        default='/root/data/video-gen-related/outputs/vbench/fastvideo_i2v',
                        help='视频所在目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认为 video_dir/vbench_output)')
    parser.add_argument('--vbench_info', type=str,
                        default='/root/data/FAR-World/assets/data/meta/vbench/vbench2_i2v_full_info.json',
                        help='VBench I2V 元信息 JSON 文件路径')
    parser.add_argument('--cache_dir', type=str,
                        default='/root/data/FAR-World/experiments/pretrained_models/vbench',
                        help='VBench 缓存目录')
    return parser.parse_args()


# I2V 评估维度
DIMENSIONS = [
    'camera_motion', 'i2v_subject', 'i2v_background',
    'subject_consistency', 'motion_smoothness', 'background_consistency',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
]

# 归一化范围
METRICS_NORMALIZATION_RANGES = {
    'subject_consistency': [0.1462, 1.0],
    'motion_smoothness': [0.706, 0.9975],
    'temporal_flickering': [0.6293, 1.0],
    'background_consistency': [0.2615, 1.0],
    'scene': [0.0, 0.8222],
    'appearance_style': [0.0009, 0.2855],
    'temporal_style': [0.0, 0.364],
    'overall_consistency': [0.0, 0.364],
    'i2v_subject': [0.1462, 1.0],
    'i2v_background': [0.2615, 1.0],
    'dynamic_degree': [0.0, 1.0],
    'aesthetic_quality': [0.0, 1.0],
    'imaging_quality': [0.0, 1.0],
    'camera_motion': [0.0, 1.0],
}


def norm(metric, key):
    range_ = METRICS_NORMALIZATION_RANGES.get(key, [0.0, 1.0])
    metric = max(metric, range_[0])
    metric = min(metric, range_[1])
    metric = (metric - range_[0]) / (range_[1] - range_[0])
    return metric


def main():
    args = parse_args()

    # 设置目录
    video_dir = args.video_dir
    output_dir = args.output_dir or os.path.join(video_dir, 'vbench_output')
    cache_dir = args.cache_dir
    vbench_info = args.vbench_info

    # 设置缓存目录
    os.environ['VBENCH_CACHE_DIR'] = cache_dir

    from vbench2_beta_i2v import VBenchI2V
    from vbench.distributed import dist_init, get_rank, get_world_size, barrier

    # 初始化分布式
    dist_init()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        video_count = len([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        print(f"=" * 60)
        print(f"VBench I2V 评估 - {world_size} GPU 并行")
        print(f"视频目录: {video_dir}")
        print(f"视频数量: {video_count}")
        print(f"输出目录: {output_dir}")
        print(f"缓存目录: {cache_dir}")
        print(f"评估维度: {len(DIMENSIONS)} 个")
        print(f"=" * 60)

    barrier()

    # 初始化 VBench I2V (位置参数: device, full_json_dir, output_path)
    evaluator = VBenchI2V(device, vbench_info, output_dir)

    # 收集所有评估结果
    eval_info_dict = {}

    for i, dimension in enumerate(DIMENSIONS):
        if rank == 0:
            print(f"\n[{i+1}/{len(DIMENSIONS)}] 评估维度: {dimension}")

        # 检查是否已有结果
        result_path = os.path.join(output_dir, f'{dimension}_eval_results.json')
        if os.path.exists(result_path):
            if rank == 0:
                with open(result_path, 'r') as f:
                    result = json.load(f)
                    eval_info_dict[dimension] = result[dimension][0]
                    print(f"  -> 已有结果: {eval_info_dict[dimension]:.4f}")
        else:
            # 运行评估
            result = evaluator.evaluate(
                videos_path=video_dir,
                name=dimension,
                dimension_list=[dimension],
                resolution='480p',
                local=True,
            )
            if rank == 0:
                eval_info_dict[dimension] = result[dimension]
                print(f"  -> 评估完成: {result[dimension]:.4f}")

        barrier()

    # 计算总分 (仅 rank 0)
    if rank == 0:
        # 重新加载所有结果
        for dimension in DIMENSIONS:
            result_path = os.path.join(output_dir, f'{dimension}_eval_results.json')
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result = json.load(f)
                    eval_info_dict[dimension] = result[dimension][0]

        # 计算 quality_score
        quality_score = (
            norm(eval_info_dict['subject_consistency'], 'subject_consistency') +
            norm(eval_info_dict['background_consistency'], 'background_consistency') +
            norm(eval_info_dict['motion_smoothness'], 'motion_smoothness') +
            norm(eval_info_dict['dynamic_degree'], 'dynamic_degree') * 0.5 +
            norm(eval_info_dict['aesthetic_quality'], 'aesthetic_quality') +
            norm(eval_info_dict['imaging_quality'], 'imaging_quality')
        ) / 5.5

        # 计算 i2v_score
        i2v_score = (
            norm(eval_info_dict['i2v_subject'], 'i2v_subject') +
            norm(eval_info_dict['i2v_background'], 'i2v_background') +
            norm(eval_info_dict['camera_motion'], 'camera_motion') * 0.1
        ) / 2.1

        # 计算 overall_score
        overall_score = 0.5 * i2v_score + 0.5 * quality_score

        eval_info_dict['quality_score'] = quality_score
        eval_info_dict['i2v_score'] = i2v_score
        eval_info_dict['overall_score'] = overall_score

        # 保存最终结果
        final_result_path = os.path.join(output_dir, 'final_results.json')
        with open(final_result_path, 'w') as f:
            json.dump(eval_info_dict, f, indent=2)

        # 打印结果
        print("\n" + "=" * 60)
        print("VBench I2V 评估结果")
        print("=" * 60)
        print("\n各维度得分 (归一化):")
        for dim in DIMENSIONS:
            raw = eval_info_dict[dim]
            normalized = norm(raw, dim)
            print(f"  {dim:25s}: {normalized:.4f} (raw: {raw:.4f})")
        print("\n综合得分:")
        print(f"  Quality Score:  {quality_score:.4f}")
        print(f"  I2V Score:      {i2v_score:.4f}")
        print(f"  Overall Score:  {overall_score:.4f}")
        print("=" * 60)
        print(f"\n结果已保存到: {final_result_path}")

    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
