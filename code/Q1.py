"""附件1裂缝识别四阶段流水线。

阶段一：合成裂缝监督。
阶段二：异常修复。
阶段三：Teacher-Student 自训练。
阶段四：独立真实审核与最终评价。

本文件按阶段逐步扩展。所有训练数据、指标和图像均由程序实际生成，
不使用已经废弃的旧人工 Mask，也不会覆盖附件中的原始 JPG。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Windows 中文字体设置；若系统缺少该字体，Matplotlib 会自动回退。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class 项目配置:
    """集中保存可复现参数，避免在函数内部散落魔法数字。"""

    项目根目录: Path = Path(__file__).resolve().parents[1]
    随机种子: int = 20260821
    图块尺寸: int = 128
    批大小: int = 4
    阶段一迭代次数: int = 120
    阶段一验证样本数: int = 48
    学习率: float = 1.0e-3
    二值阈值: float = 0.50
    有效区顶部: int = 24

    @property
    def 输入目录(self) -> Path:
        return self.项目根目录 / "附件1"

    @property
    def 输出根目录(self) -> Path:
        return self.项目根目录 / "output" / "Q1_result"

    @property
    def 阶段一目录(self) -> Path:
        return self.输出根目录 / "阶段一_合成裂缝监督"


配置 = 项目配置()


def 设置随机种子(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def 自然排序键(path: Path) -> list[object]:
    """使图1-2排在图1-10之前。"""

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def 获取输入图像() -> list[Path]:
    """读取附件1全部 JPG，并验证数量、尺寸和色彩模式。"""

    paths = sorted(配置.输入目录.glob("*.jpg"), key=自然排序键)
    if len(paths) != 10:
        raise RuntimeError(f"附件1应有10张JPG，实际找到{len(paths)}张。")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (244, 1350) or image.mode != "RGB":
                raise RuntimeError(
                    f"{path.name}规格异常：size={image.size}, mode={image.mode}。"
                )
    return paths


def 读取RGB(path: Path) -> np.ndarray:
    """以0～1浮点数组读取RGB，不修改原文件。"""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def 循环裁剪图块(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """横向按360°循环取样，纵向采用普通窗口。"""

    height, width = image.shape[:2]
    y0 = max(0, min(y0, height - size))
    x_indices = (np.arange(size) + x0) % width
    return image[y0 : y0 + size, x_indices].copy()


def 生成裂缝骨架(
    rng: np.random.Generator, size: int, kind: str
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """生成正弦、随机曲线或近竖直裂缝，并返回精确二值Mask。"""

    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    width = int(rng.integers(1, 5))
    points: list[tuple[int, int]] = []

    if kind == "周期正弦裂缝":
        center = float(rng.uniform(size * 0.25, size * 0.75))
        amplitude = float(rng.uniform(size * 0.08, size * 0.32))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        for x in range(-4, size + 5):
            y = center + amplitude * math.sin(2.0 * math.pi * x / size + phase)
            y += float(rng.normal(0.0, 0.6))
            points.append((x, int(round(y))))
    elif kind == "近竖直裂缝":
        center = float(rng.uniform(size * 0.25, size * 0.75))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        for y in range(-4, size + 5):
            x = center + 0.08 * y + 5.0 * math.sin(y / 25.0 + phase)
            x += float(rng.normal(0.0, 0.7))
            points.append((int(round(x)), y))
        amplitude = 0.0
    else:
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, max(1, size // 5)))
        points.append((x, y))
        while y < size + 8:
            x = int(np.clip(x + rng.normal(0.0, 5.0), -8, size + 8))
            y += int(rng.integers(4, 10))
            points.append((x, y))
        amplitude = 0.0

    draw.line(points, fill=255, width=width, joint="curve")

    # 部分样本增加短分支，模拟裂缝分叉但避免生成密集树状伪影。
    branch_count = 0
    if rng.random() < 0.35 and len(points) > 8:
        branch_count = 1
        anchor = points[int(rng.integers(len(points) // 3, 2 * len(points) // 3))]
        length = int(rng.integers(size // 8, size // 3))
        direction = float(rng.choice([-1.0, 1.0]))
        branch = [anchor]
        for step in range(1, length, 4):
            branch.append(
                (
                    int(anchor[0] + direction * step + rng.normal(0.0, 1.5)),
                    int(anchor[1] + 0.6 * step + rng.normal(0.0, 1.0)),
                )
            )
        draw.line(branch, fill=255, width=max(1, width - 1), joint="curve")

    mask = (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.float32)
    return mask, {
        "裂缝类型": kind,
        "裂缝宽度像素": width,
        "正弦振幅像素": round(amplitude, 3),
        "分支数量": branch_count,
    }


def 加入困难负样本(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    """加入不计入Mask的条纹或层理，训练模型不要把所有线都当裂缝。"""

    output = image.copy()
    size = output.shape[0]
    negative_type = "无"
    if rng.random() < 0.45:
        negative_type = str(rng.choice(["水平层理", "竖向成像条纹", "局部阴影"]))
        if negative_type == "水平层理":
            y = int(rng.integers(8, size - 8))
            thickness = int(rng.integers(2, 7))
            delta = float(rng.uniform(-0.10, 0.10))
            output[max(0, y - thickness) : min(size, y + thickness)] += delta
        elif negative_type == "竖向成像条纹":
            x = int(rng.integers(8, size - 8))
            thickness = int(rng.integers(2, 6))
            delta = float(rng.uniform(-0.08, 0.08))
            output[:, max(0, x - thickness) : min(size, x + thickness)] += delta
        else:
            yy, xx = np.mgrid[:size, :size]
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
            sigma = rng.uniform(size * 0.15, size * 0.35)
            field = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
            output += field[..., None] * float(rng.uniform(-0.12, 0.12))
    return np.clip(output, 0.0, 1.0), negative_type


def 合成裂缝样本(
    background: np.ndarray,
    rng: np.random.Generator,
    force_positive: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """在真实岩壁图块上合成裂缝，背景保持为真实附件纹理。"""

    image, negative_type = 加入困难负样本(background, rng)
    positive = rng.random() < 0.70 if force_positive is None else force_positive
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    metadata: dict[str, object] = {
        "是否含裂缝": int(positive),
        "困难负样本": negative_type,
    }
    if not positive:
        noise = rng.normal(0.0, rng.uniform(0.002, 0.012), image.shape)
        return np.clip(image + noise, 0.0, 1.0).astype(np.float32), mask, metadata

    kind = str(rng.choice(["周期正弦裂缝", "随机曲线裂缝", "近竖直裂缝"], p=[0.5, 0.3, 0.2]))
    mask, geometry = 生成裂缝骨架(rng, image.shape[0], kind)
    metadata.update(geometry)

    # 裂缝既可能比围岩暗，也可能因填充物而更亮，避免模型只学习黑线。
    local_mean = image.mean(axis=(0, 1), keepdims=True)
    if rng.random() < 0.62:
        target = local_mean * float(rng.uniform(0.18, 0.62))
        appearance = "暗色开口"
    else:
        target = np.clip(local_mean + rng.uniform(0.18, 0.48), 0.0, 1.0)
        target = target * np.array([1.0, rng.uniform(0.88, 1.04), rng.uniform(0.65, 0.95)])
        appearance = "浅色填充"
    texture = target + rng.normal(0.0, 0.035, image.shape)
    alpha = mask[..., None] * float(rng.uniform(0.72, 0.96))
    image = image * (1.0 - alpha) + texture * alpha
    image += rng.normal(0.0, rng.uniform(0.003, 0.015), image.shape)
    metadata["裂缝外观"] = appearance
    return np.clip(image, 0.0, 1.0).astype(np.float32), mask, metadata


class 在线合成批次:
    """从指定原图持续生成随机训练批次，不在磁盘复制大量样本。"""

    def __init__(self, image_paths: Sequence[Path], seed: int):
        self.paths = list(image_paths)
        self.images = [读取RGB(path) for path in self.paths]
        self.rng = np.random.default_rng(seed)

    def 获取图块(self) -> tuple[np.ndarray, str, int, int]:
        index = int(self.rng.integers(0, len(self.images)))
        image = self.images[index]
        y0 = int(
            self.rng.integers(
                配置.有效区顶部,
                max(配置.有效区顶部 + 1, image.shape[0] - 配置.图块尺寸 + 1),
            )
        )
        x0 = int(self.rng.integers(0, image.shape[1]))
        patch = 循环裁剪图块(image, x0, y0, 配置.图块尺寸)
        return patch, self.paths[index].stem, x0, y0

    def 生成批次(self, batch_size: int) -> tuple[Tensor, Tensor]:
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for _ in range(batch_size):
            patch, _, _, _ = self.获取图块()
            synthetic, mask, _ = 合成裂缝样本(patch, self.rng)
            images.append(synthetic.transpose(2, 0, 1))
            masks.append(mask[None, ...])
        return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


class 双卷积块(nn.Module):
    """两次卷积配合GroupNorm，适合CPU小批量训练。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = 4 if out_channels >= 4 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class 轻量裂缝UNet(nn.Module):
    """为小样本和CPU环境设计的轻量像素级分割网络。"""

    def __init__(self, in_channels: int = 3, base: int = 8):
        super().__init__()
        self.enc1 = 双卷积块(in_channels, base)
        self.enc2 = 双卷积块(base, base * 2)
        self.enc3 = 双卷积块(base * 2, base * 4)
        self.bridge = 双卷积块(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = 双卷积块(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = 双卷积块(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = 双卷积块(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def 分割损失(logits: Tensor, targets: Tensor) -> Tensor:
    """Focal BCE与Dice联合损失，兼顾稀少裂缝像素和区域重叠。"""

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha = 0.78 * targets + 0.22 * (1.0 - targets)
    focal = (alpha * (1.0 - pt).pow(2.0) * bce).mean()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return focal + 0.8 * dice


def 混淆统计(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, int]:
    """按像素累计TP、FP、FN、TN。"""

    predictions = probabilities >= threshold
    labels = targets >= 0.5
    return {
        "TP": int(np.logical_and(predictions, labels).sum()),
        "FP": int(np.logical_and(predictions, ~labels).sum()),
        "FN": int(np.logical_and(~predictions, labels).sum()),
        "TN": int(np.logical_and(~predictions, ~labels).sum()),
    }


def 由混淆统计计算指标(counts: dict[str, int]) -> dict[str, float]:
    """由混淆矩阵计算Precision、Recall、F1、IoU和Accuracy。"""

    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    eps = 1.0e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "精确率Precision": precision,
        "召回率Recall": recall,
        "F1分数": f1,
        "交并比IoU": iou,
        "像素准确率Accuracy": accuracy,
        "特异度Specificity": specificity,
    }


def 保存JSON(data: object, path: Path) -> None:
    """以UTF-8中文格式保存JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def 保存训练日志(rows: Sequence[dict[str, float | int]], path: Path) -> None:
    """保存可复算的逐迭代损失CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["迭代次数", "训练损失"])
        writer.writeheader()
        writer.writerows(rows)


def 绘制损失曲线(rows: Sequence[dict[str, float | int]], path: Path) -> None:
    """绘制迭代次数—训练损失折线图。"""

    x = [int(row["迭代次数"]) for row in rows]
    y = [float(row["训练损失"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    ax.plot(x, y, color="#2F6B9A", linewidth=1.6)
    if len(y) >= 10:
        kernel = np.ones(10) / 10.0
        smooth = np.convolve(y, kernel, mode="valid")
        ax.plot(x[9:], smooth, color="#D39C2C", linewidth=2.2, label="10次滑动平均")
        ax.legend(frameon=False)
    ax.set_title("阶段一合成监督训练损失")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("Focal BCE + Dice损失")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.text(0.5, 0.01, "数据来源：附件1真实岩壁背景上的在线合成裂缝", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制混淆矩阵(counts: dict[str, int], path: Path) -> None:
    """输出合成验证集像素混淆矩阵。"""

    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    row_sum = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / row_sum
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}\n{normalized[row, col]:.2%}",
                ha="center",
                va="center",
                color="white" if normalized[row, col] > 0.55 else "#1F2933",
                fontsize=12,
            )
    ax.set_xticks([0, 1], labels=["预测背景", "预测裂缝"])
    ax.set_yticks([0, 1], labels=["真实背景", "真实裂缝"])
    ax.set_title("阶段一合成验证集混淆矩阵")
    ax.set_xlabel("模型预测")
    ax.set_ylabel("合成真值")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="按真实类别归一化比例")
    fig.text(0.5, 0.01, "注意：这是合成真值指标，不代表真实钻孔裂缝精度", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制指标柱状图(metrics: dict[str, float], path: Path) -> None:
    """比较同一合成验证集上的核心分割指标。"""

    names = ["精确率", "召回率", "F1", "IoU", "特异度"]
    values = [
        metrics["精确率Precision"],
        metrics["召回率Recall"],
        metrics["F1分数"],
        metrics["交并比IoU"],
        metrics["特异度Specificity"],
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    bars = ax.bar(names, values, color="#2F6B9A", edgecolor="#203A4F", linewidth=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("阶段一合成验证集分割指标")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    fig.text(0.5, 0.01, "阈值=0.50；评价对象为独立背景图上的在线合成裂缝", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 加载中文字体(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """为Pillow输出图加载中文字体。"""

    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def 转为PIL(array: np.ndarray) -> Image.Image:
    """把0～1浮点RGB转换为Pillow图像。"""

    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def 保存合成对比图(
    background: np.ndarray,
    synthetic: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    metadata: dict[str, object],
    path: Path,
) -> None:
    """保存原始图块、合成图、真值Mask和模型概率的中文对比图。"""

    size = background.shape[0]
    scale = 2
    panel = size * scale
    label_height = 66
    gap = 12
    canvas = Image.new("RGB", (4 * panel + 5 * gap, panel + label_height), "white")
    font = 加载中文字体(18)
    draw = ImageDraw.Draw(canvas)
    mask_rgb = np.repeat(mask[..., None], 3, axis=2)
    prob_rgb = np.zeros_like(mask_rgb)
    prob_rgb[..., 0] = probability
    prob_rgb[..., 1] = probability * 0.35
    panels = [background, synthetic, mask_rgb, prob_rgb]
    labels = ["真实背景图块", "合成裂缝图", "合成真值Mask", "模型裂缝概率"]
    for index, (array, label) in enumerate(zip(panels, labels)):
        left = gap + index * (panel + gap)
        image = 转为PIL(array).resize((panel, panel), Image.Resampling.NEAREST if index == 2 else Image.Resampling.BILINEAR)
        canvas.paste(image, (left, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((left + (panel - (bbox[2] - bbox[0])) // 2, panel + 8), label, font=font, fill="#18222B")
    detail = f"类型：{metadata.get('裂缝类型', '无裂缝')}；困难负样本：{metadata.get('困难负样本', '无')}"
    draw.text((gap, panel + 36), detail, font=加载中文字体(14), fill="#4B5563")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def 验证合成模型(
    model: nn.Module,
    generator: 在线合成批次,
    sample_count: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """在未参与训练的原图背景上生成固定数量合成样本并计算指标。"""

    model.eval()
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    remaining = sample_count
    with torch.no_grad():
        while remaining > 0:
            batch = min(配置.批大小, remaining)
            images, masks = generator.生成批次(batch)
            probabilities = torch.sigmoid(model(images)).cpu().numpy()
            batch_counts = 混淆统计(probabilities, masks.numpy(), 配置.二值阈值)
            for key in counts:
                counts[key] += batch_counts[key]
            remaining -= batch
    return counts, 由混淆统计计算指标(counts)


def 运行阶段一() -> None:
    """执行合成裂缝监督训练、验证、可视化和结果落盘。"""

    设置随机种子(配置.随机种子)
    paths = 获取输入图像()
    train_paths = paths[:8]
    validation_paths = paths[8:]
    output = 配置.阶段一目录
    output.mkdir(parents=True, exist_ok=True)

    train_generator = 在线合成批次(train_paths, 配置.随机种子 + 11)
    validation_generator = 在线合成批次(validation_paths, 配置.随机种子 + 29)
    model = 轻量裂缝UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=配置.学习率, weight_decay=1.0e-4)
    history: list[dict[str, float | int]] = []

    model.train()
    for iteration in range(1, 配置.阶段一迭代次数 + 1):
        images, masks = train_generator.生成批次(配置.批大小)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = 分割损失(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append({"迭代次数": iteration, "训练损失": float(loss.item())})
        if iteration == 1 or iteration % 20 == 0:
            print(f"阶段一迭代 {iteration:03d}/{配置.阶段一迭代次数}，损失={loss.item():.6f}")

    checkpoint_path = output / "阶段一_合成监督模型.pt"
    torch.save(
        {
            "模型参数": model.state_dict(),
            "配置": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(配置).items()},
            "训练原图": [path.name for path in train_paths],
            "验证原图": [path.name for path in validation_paths],
        },
        checkpoint_path,
    )

    counts, metrics = 验证合成模型(model, validation_generator, 配置.阶段一验证样本数)
    report = {
        "阶段": "阶段一_合成裂缝监督",
        "说明": "指标来自未参与训练的真实岩壁背景上的合成裂缝，不等同于真实裂缝精度。",
        "训练原图": [path.name for path in train_paths],
        "验证原图": [path.name for path in validation_paths],
        "验证合成样本数": 配置.阶段一验证样本数,
        "二值阈值": 配置.二值阈值,
        "混淆统计": counts,
        "指标": metrics,
    }
    保存JSON(report, output / "阶段一_合成验证指标.json")
    保存训练日志(history, output / "阶段一_训练损失.csv")
    绘制损失曲线(history, output / "阶段一_迭代次数与损失曲线.png")
    绘制混淆矩阵(counts, output / "阶段一_合成验证混淆矩阵.png")
    绘制指标柱状图(metrics, output / "阶段一_合成验证指标柱状图.png")

    # 为每张原图保存一张真实背景与合成修改的对比图，文件名全部使用中文。
    sample_dir = output / "合成裂缝对比图"
    model.eval()
    for index, path in enumerate(paths, start=1):
        image = 读取RGB(path)
        rng = np.random.default_rng(配置.随机种子 + 1000 + index)
        y0 = int(rng.integers(配置.有效区顶部, image.shape[0] - 配置.图块尺寸))
        x0 = int(rng.integers(0, image.shape[1]))
        background = 循环裁剪图块(image, x0, y0, 配置.图块尺寸)
        synthetic, mask, metadata = 合成裂缝样本(background, rng, force_positive=True)
        with torch.no_grad():
            tensor = torch.from_numpy(synthetic.transpose(2, 0, 1)[None])
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        metadata.update({"原图": path.name, "起始横坐标": x0, "起始纵坐标": y0})
        保存合成对比图(
            background,
            synthetic,
            mask,
            probability,
            metadata,
            sample_dir / f"{path.stem}_合成裂缝监督对比图.png",
        )

    保存JSON(
        {
            "随机种子": 配置.随机种子,
            "PyTorch版本": torch.__version__,
            "运行设备": "CPU",
            "输入图像数量": len(paths),
            "输出目录": str(output),
        },
        output / "阶段一_运行清单.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def 构建命令行() -> argparse.ArgumentParser:
    """构建分阶段命令行入口。"""

    parser = argparse.ArgumentParser(description="附件1裂缝识别四阶段流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("阶段一", help="运行合成裂缝监督阶段")
    return parser


def main() -> None:
    args = 构建命令行().parse_args()
    if args.command == "阶段一":
        运行阶段一()


if __name__ == "__main__":
    main()
