"""
裁剪投影数据 .raw 文件的黑边

用法：
    python crop_proj_raw.py <输入.raw> [输出.raw] [--top N] [--bottom N]

示例：
    python crop_proj_raw.py proj_no_pbi.raw proj_cropped.raw
    python crop_proj_raw.py proj_no_pbi.raw --top 8 --bottom 8
"""

import argparse
import numpy as np
from pathlib import Path

PROJ_SHAPE = (720, 128, 512)  # (views, rows, channels)
DTYPE = np.float32


def main():
    parser = argparse.ArgumentParser(description="裁剪投影数据 .raw 的上下黑边")
    parser.add_argument("input", type=str, help="输入的 .raw 文件路径")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="输出的 .raw 文件路径（默认: 输入文件_cropped.raw）")
    parser.add_argument("--top", type=int, default=8,
                        help="上方裁剪行数（默认: 8）")
    parser.add_argument("--bottom", type=int, default=8,
                        help="下方裁剪行数（默认: 8）")
    parser.add_argument("--info", action="store_true",
                        help="只打印统计信息，不执行裁剪")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return

    # 自动推断输出文件名
    if args.output is None and not args.info:
        stem = input_path.stem  # 去掉 .raw 后缀
        output_path = input_path.with_name(f"{stem}_cropped.raw")
    elif args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = None

    # 读取数据
    print(f"📖 读取: {input_path}")
    data = np.fromfile(input_path, dtype=DTYPE).reshape(PROJ_SHAPE)
    print(f"   形状: {data.shape}")
    print(f"   数值范围: [{data.min():.6f}, {data.max():.6f}]")

    # 分析每行的均值/方差
    row_means = data.mean(axis=(0, 2))     # (128,) 所有视角每个探测器行的平均
    row_stds = data.std(axis=(0, 2))       # (128,) 所有视角每个探测器行的标准差

    print(f"\n📊 每行统计（所有 {PROJ_SHAPE[0]} 个视角平均）:")
    print(f"   {'行号':>4s}  {'均值':>12s}  {'标准差':>12s}  {'判断'}")
    print(f"   {'----':>4s}  {'------------':>12s}  {'------------':>12s}  {'----'}")

    # 找有效行（均值 > 均值中心区域 1% 的行）
    center_region = slice(30, 100)
    center_mean = row_means[center_region].mean()
    threshold = center_mean * 0.01
    valid_flags = row_means > threshold

    valid_indices = np.where(valid_flags)[0]
    if len(valid_indices) > 0:
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
    else:
        first_valid = last_valid = 0

    for i in range(128):
        is_valid = valid_flags[i]
        tag = "✅" if is_valid else ("⬛" if i < 30 or i >= 100 else "⚠️")
        print(f"   {i:>4d}  {row_means[i]:>12.6f}  {row_stds[i]:>12.6f}  {tag}")

    print(f"\n📐 有效行范围: {first_valid} ~ {last_valid}（共 {last_valid - first_valid + 1} 行）")
    print(f"✂️  拟裁剪: 上 {args.top} 行 + 下 {args.bottom} 行 → 输出高度 {128 - args.top - args.bottom}")

    # 检查裁剪是否覆盖黑边
    if first_valid >= args.top:
        print(f"✅ 上方裁剪 {args.top} 行覆盖了黑边（有效从第 {first_valid} 行开始）")
    else:
        print(f"⚠️  上方有效信号从第 {first_valid} 行开始，裁剪 {args.top} 行会切到有效数据！")

    if last_valid < 128 - args.bottom:
        print(f"✅ 下方裁剪 {args.bottom} 行覆盖了黑边（有效到第 {last_valid} 行结束）")
    else:
        print(f"⚠️  下方有效信号到第 {last_valid} 行，裁剪 {args.bottom} 行会切到有效数据！")

    if args.info:
        print("\nℹ️  --info 模式，不执行裁剪")
        return

    # 执行裁剪
    cropped = data[:, args.top:128 - args.bottom, :]  # (720, 128-top-bottom, 512)
    new_shape = cropped.shape
    print(f"\n✂️  裁剪完成: {data.shape} → {new_shape}")

    # 保存
    cropped.tofile(output_path)
    file_size = output_path.stat().st_size
    expected_size = np.prod(new_shape) * np.dtype(DTYPE).itemsize
    print(f"💾 保存: {output_path}")
    print(f"   形状: {new_shape}")
    print(f"   大小: {file_size / 1024 / 1024:.2f} MB")
    print(f"   验证: {'✅ 文件大小正确' if file_size == expected_size else '❌ 大小异常'}")

    # 保存后快速验证读取
    verify = np.fromfile(output_path, dtype=DTYPE).reshape(new_shape)
    print(f"   重读验证: shape={verify.shape}, range=[{verify.min():.6f}, {verify.max():.6f}]  ✅")


if __name__ == "__main__":
    main()
