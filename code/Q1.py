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
import hashlib
import json
import math
import random
import re
import shutil
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
    阶段二修复迭代次数: int = 100
    阶段二引导分割迭代次数: int = 80
    阶段三迭代次数: int = 140
    Teacher_EMA系数: float = 0.995
    真实审核窗口高度: int = 160
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

    @property
    def 阶段二目录(self) -> Path:
        return self.输出根目录 / "阶段二_异常修复"

    @property
    def 阶段三目录(self) -> Path:
        return self.输出根目录 / "阶段三_Teacher-Student自训练"

    @property
    def 阶段四目录(self) -> Path:
        return self.输出根目录 / "阶段四_独立真实无标签审核"


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

    def 生成修复批次(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """生成合成裂缝图、对应正常背景和精确Mask。"""

        synthetic_images: list[np.ndarray] = []
        clean_images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for _ in range(batch_size):
            clean, _, _, _ = self.获取图块()
            synthetic, mask, _ = 合成裂缝样本(clean, self.rng)
            synthetic_images.append(synthetic.transpose(2, 0, 1))
            clean_images.append(clean.transpose(2, 0, 1))
            masks.append(mask[None, ...])
        return (
            torch.from_numpy(np.stack(synthetic_images)),
            torch.from_numpy(np.stack(clean_images)),
            torch.from_numpy(np.stack(masks)),
        )


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


class 轻量修复UNet(nn.Module):
    """把含合成裂缝的图块恢复为无裂缝岩壁背景。"""

    def __init__(self, base: int = 8):
        super().__init__()
        self.enc1 = 双卷积块(3, base)
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
        self.head = nn.Conv2d(base, 3, 1)
        # 零初始化使网络训练开始时严格复制输入，之后只学习必要的局部残差。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        residual = 0.35 * torch.tanh(self.head(d1))
        return torch.clamp(x + residual, 0.0, 1.0)


def 修复损失(repaired: Tensor, clean: Tensor, masks: Tensor) -> tuple[Tensor, dict[str, float]]:
    """联合Mask内修复、Mask外恒等和梯度保持损失。"""

    mask3 = masks.expand_as(repaired)
    inside = (torch.abs(repaired - clean) * mask3).sum() / (mask3.sum() + 1.0)
    outside_mask = 1.0 - mask3
    outside = (torch.abs(repaired - clean) * outside_mask).sum() / (outside_mask.sum() + 1.0)
    repaired_dx = repaired[:, :, :, 1:] - repaired[:, :, :, :-1]
    clean_dx = clean[:, :, :, 1:] - clean[:, :, :, :-1]
    repaired_dy = repaired[:, :, 1:, :] - repaired[:, :, :-1, :]
    clean_dy = clean[:, :, 1:, :] - clean[:, :, :-1, :]
    gradient = F.l1_loss(repaired_dx, clean_dx) + F.l1_loss(repaired_dy, clean_dy)
    total = 4.0 * inside + 2.5 * outside + 0.35 * gradient
    return total, {
        "Mask内MAE": float(inside.item()),
        "Mask外MAE": float(outside.item()),
        "梯度损失": float(gradient.item()),
    }


def 构造修复引导输入(images: Tensor, repaired: Tensor) -> Tensor:
    """把RGB原图与单通道修复差异拼成四通道分割输入。"""

    residual = torch.mean(torch.abs(images - repaired), dim=1, keepdim=True)
    return torch.cat([images, residual], dim=1)


def 从三通道模型初始化四通道模型(checkpoint_path: Path) -> 轻量裂缝UNet:
    """保留阶段一全部参数，并把新增残差通道初始权重设为零。"""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["模型参数"]
    model = 轻量裂缝UNet(in_channels=4)
    target = model.state_dict()
    for key, value in source.items():
        if key == "enc1.layers.0.weight":
            target[key][:, :3] = value
            target[key][:, 3:] = 0.0
        elif key in target and target[key].shape == value.shape:
            target[key] = value
    model.load_state_dict(target)
    return model


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


def 评估修复网络(
    model: 轻量修复UNet,
    generator: 在线合成批次,
    sample_count: int,
) -> dict[str, float]:
    """在独立背景上统计修复网络的Mask内/外误差和全图PSNR。"""

    model.eval()
    inside_error = 0.0
    outside_error = 0.0
    inside_pixels = 0.0
    outside_pixels = 0.0
    squared_error = 0.0
    value_count = 0
    remaining = sample_count
    with torch.no_grad():
        while remaining > 0:
            batch = min(配置.批大小, remaining)
            synthetic, clean, masks = generator.生成修复批次(batch)
            repaired = model(synthetic)
            mask3 = masks.expand_as(repaired)
            inside_error += float((torch.abs(repaired - clean) * mask3).sum().item())
            outside_error += float((torch.abs(repaired - clean) * (1.0 - mask3)).sum().item())
            inside_pixels += float(mask3.sum().item())
            outside_pixels += float((1.0 - mask3).sum().item())
            squared_error += float(F.mse_loss(repaired, clean, reduction="sum").item())
            value_count += repaired.numel()
            remaining -= batch
    mse = squared_error / max(value_count, 1)
    return {
        "Mask内平均绝对误差": inside_error / max(inside_pixels, 1.0),
        "Mask外平均绝对误差": outside_error / max(outside_pixels, 1.0),
        "全图均方误差": mse,
        "全图峰值信噪比PSNR": -10.0 * math.log10(max(mse, 1.0e-12)),
    }


def 验证修复引导分割器(
    repair_model: 轻量修复UNet,
    segmentation_model: 轻量裂缝UNet,
    generator: 在线合成批次,
    sample_count: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """以修复残差为第四通道，计算相同定义的合成验证指标。"""

    repair_model.eval()
    segmentation_model.eval()
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    remaining = sample_count
    with torch.no_grad():
        while remaining > 0:
            batch = min(配置.批大小, remaining)
            images, _, masks = generator.生成修复批次(batch)
            repaired = repair_model(images)
            probabilities = torch.sigmoid(
                segmentation_model(构造修复引导输入(images, repaired))
            ).cpu().numpy()
            current = 混淆统计(probabilities, masks.numpy(), 配置.二值阈值)
            for key in counts:
                counts[key] += current[key]
            remaining -= batch
    return counts, 由混淆统计计算指标(counts)


def 绘制双损失曲线(rows: Sequence[dict[str, float | int]], path: Path) -> None:
    """绘制修复网络总损失、Mask内误差和Mask外误差。"""

    x = [int(row["迭代次数"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=150)
    ax.plot(x, [float(row["修复总损失"]) for row in rows], color="#2F6B9A", label="修复总损失")
    ax.plot(x, [float(row["Mask内MAE"]) for row in rows], color="#D39C2C", label="Mask内MAE")
    ax.plot(x, [float(row["Mask外MAE"]) for row in rows], color="#6B7280", label="Mask外MAE")
    ax.set_title("阶段二异常修复训练曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("损失或平均绝对误差")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, ncol=3)
    fig.text(0.5, 0.01, "Mask内衡量裂缝清除；Mask外衡量正常岩壁保持", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制阶段指标对比(
    baseline: dict[str, float], guided: dict[str, float], path: Path
) -> None:
    """用分组柱状图比较阶段一与修复引导分割指标。"""

    names = ["精确率", "召回率", "F1", "IoU"]
    keys = ["精确率Precision", "召回率Recall", "F1分数", "交并比IoU"]
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=150)
    bars1 = ax.bar(x - width / 2, [baseline[key] for key in keys], width, label="阶段一合成监督", color="#AFC8DA", edgecolor="#2F6B9A")
    bars2 = ax.bar(x + width / 2, [guided[key] for key in keys], width, label="阶段二修复引导", color="#D39C2C", edgecolor="#8A6415")
    ax.set_xticks(x, names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("异常修复证据加入前后的合成验证指标")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{bar.get_height():.3f}", ha="center", fontsize=8)
    fig.text(0.5, 0.01, "相同验证原图、样本数与二值阈值；仅比较模型输入证据变化", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 滑窗修复整图(model: 轻量修复UNet, image: np.ndarray) -> np.ndarray:
    """横向循环、纵向重叠滑窗修复整张244×1350原图。"""

    size = 配置.图块尺寸
    stride = size // 2
    height, width = image.shape[:2]
    y_starts = list(range(0, max(1, height - size + 1), stride))
    if not y_starts or y_starts[-1] != height - size:
        y_starts.append(height - size)
    x_starts = list(range(0, width, stride))
    accumulator = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                patch = 循环裁剪图块(image, x0, y0, size)
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)[None])
                repaired = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
                for local_x in range(size):
                    global_x = (x0 + local_x) % width
                    accumulator[y0 : y0 + size, global_x] += repaired[:, local_x]
                    weights[y0 : y0 + size, global_x] += 1.0
    result = accumulator / np.maximum(weights, 1.0)
    # 标题/方位标记区不参与修复，保持与原图完全一致。
    result[: 配置.有效区顶部] = image[: 配置.有效区顶部]
    return np.clip(result, 0.0, 1.0)


def 残差着色(residual: np.ndarray) -> np.ndarray:
    """将0～1残差映射为黑—橙—白，便于观察修复修改位置。"""

    value = np.clip(residual * 6.0, 0.0, 1.0)
    rgb = np.zeros((*value.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(value * 1.5, 0.0, 1.0)
    rgb[..., 1] = np.clip((value - 0.25) * 1.2, 0.0, 0.75)
    rgb[..., 2] = np.clip((value - 0.65) * 2.0, 0.0, 1.0)
    return rgb


def 保存修复整图对比(original: np.ndarray, repaired: np.ndarray, path: Path) -> None:
    """输出原图、修复图和差异放大图三联图。"""

    residual = np.mean(np.abs(original - repaired), axis=2)
    panels = [转为PIL(original), 转为PIL(repaired), 转为PIL(残差着色(residual))]
    labels = ["原始钻孔图像", "异常修复图像", "修复差异热力图（放大6倍）"]
    panel_width = 244
    panel_height = 1350
    gap = 22
    label_height = 58
    canvas = Image.new("RGB", (3 * panel_width + 4 * gap, panel_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = 加载中文字体(18)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = gap + index * (panel_width + gap)
        canvas.paste(panel, (left, 0))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((left + (panel_width - (box[2] - box[0])) // 2, panel_height + 14), label, font=font, fill="#18222B")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def 运行阶段二() -> None:
    """训练异常修复网络、修复引导分割器并输出全图对比结果。"""

    设置随机种子(配置.随机种子 + 200)
    paths = 获取输入图像()
    train_paths, validation_paths = paths[:8], paths[8:]
    output = 配置.阶段二目录
    output.mkdir(parents=True, exist_ok=True)
    train_generator = 在线合成批次(train_paths, 配置.随机种子 + 211)
    repair_model = 轻量修复UNet()
    optimizer = torch.optim.AdamW(repair_model.parameters(), lr=配置.学习率, weight_decay=1.0e-4)
    repair_history: list[dict[str, float | int]] = []

    repair_model.train()
    for iteration in range(1, 配置.阶段二修复迭代次数 + 1):
        synthetic, clean, masks = train_generator.生成修复批次(配置.批大小)
        optimizer.zero_grad(set_to_none=True)
        repaired = repair_model(synthetic)
        loss, components = 修复损失(repaired, clean, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(repair_model.parameters(), 5.0)
        optimizer.step()
        repair_history.append({"迭代次数": iteration, "修复总损失": float(loss.item()), **components})
        if iteration == 1 or iteration % 20 == 0:
            print(f"阶段二修复迭代 {iteration:03d}/{配置.阶段二修复迭代次数}，损失={loss.item():.6f}")

    repair_checkpoint = output / "阶段二_异常修复模型.pt"
    torch.save({"模型参数": repair_model.state_dict(), "配置": asdict(配置)}, repair_checkpoint)
    repair_metrics = 评估修复网络(
        repair_model,
        在线合成批次(validation_paths, 配置.随机种子 + 229),
        配置.阶段一验证样本数,
    )

    segmentation_model = 从三通道模型初始化四通道模型(
        配置.阶段一目录 / "阶段一_合成监督模型.pt"
    )
    segmentation_optimizer = torch.optim.AdamW(segmentation_model.parameters(), lr=4.0e-4, weight_decay=1.0e-4)
    guided_history: list[dict[str, float | int]] = []
    repair_model.eval()
    segmentation_model.train()
    for iteration in range(1, 配置.阶段二引导分割迭代次数 + 1):
        images, _, masks = train_generator.生成修复批次(配置.批大小)
        with torch.no_grad():
            repaired = repair_model(images)
        segmentation_optimizer.zero_grad(set_to_none=True)
        loss = 分割损失(segmentation_model(构造修复引导输入(images, repaired)), masks)
        loss.backward()
        nn.utils.clip_grad_norm_(segmentation_model.parameters(), 5.0)
        segmentation_optimizer.step()
        guided_history.append({"迭代次数": iteration, "训练损失": float(loss.item())})
        if iteration == 1 or iteration % 20 == 0:
            print(f"阶段二引导分割迭代 {iteration:03d}/{配置.阶段二引导分割迭代次数}，损失={loss.item():.6f}")

    torch.save(
        {"模型参数": segmentation_model.state_dict(), "输入通道": 4, "配置": asdict(配置)},
        output / "阶段二_修复引导分割模型.pt",
    )
    guided_counts, guided_metrics = 验证修复引导分割器(
        repair_model,
        segmentation_model,
        在线合成批次(validation_paths, 配置.随机种子 + 229),
        配置.阶段一验证样本数,
    )
    baseline_report = json.loads(
        (配置.阶段一目录 / "阶段一_合成验证指标.json").read_text(encoding="utf-8")
    )
    report = {
        "阶段": "阶段二_异常修复",
        "说明": "修复指标与引导分割指标均基于独立背景上的合成裂缝；真实指标留待阶段四审核集。",
        "修复指标": repair_metrics,
        "阶段一分割指标": baseline_report["指标"],
        "阶段二修复引导分割混淆统计": guided_counts,
        "阶段二修复引导分割指标": guided_metrics,
    }
    保存JSON(report, output / "阶段二_异常修复与分割指标.json")
    with (output / "阶段二_修复训练日志.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(repair_history[0]))
        writer.writeheader()
        writer.writerows(repair_history)
    保存训练日志(guided_history, output / "阶段二_引导分割训练损失.csv")
    绘制双损失曲线(repair_history, output / "阶段二_异常修复迭代与损失曲线.png")
    绘制损失曲线(guided_history, output / "阶段二_引导分割迭代与损失曲线.png")
    绘制混淆矩阵(guided_counts, output / "阶段二_修复引导分割混淆矩阵.png")
    绘制阶段指标对比(
        baseline_report["指标"], guided_metrics, output / "阶段一与阶段二指标对比图.png"
    )

    repaired_dir = output / "原图异常修复结果"
    comparison_dir = output / "原图修复对比图"
    residual_dir = output / "原图修复差异图"
    for path in paths:
        original = 读取RGB(path)
        repaired = 滑窗修复整图(repair_model, original)
        residual = np.mean(np.abs(original - repaired), axis=2)
        repaired_dir.mkdir(parents=True, exist_ok=True)
        residual_dir.mkdir(parents=True, exist_ok=True)
        转为PIL(repaired).save(repaired_dir / f"{path.stem}_异常修复图.png")
        转为PIL(残差着色(residual)).save(residual_dir / f"{path.stem}_修复差异热力图.png")
        保存修复整图对比(original, repaired, comparison_dir / f"{path.stem}_异常修复三联对比图.png")

    print(json.dumps(report, ensure_ascii=False, indent=2))


def 加载阶段二模型() -> tuple[轻量修复UNet, 轻量裂缝UNet]:
    """加载阶段二冻结修复网络和四通道分割网络。"""

    repair = 轻量修复UNet()
    repair_checkpoint = torch.load(
        配置.阶段二目录 / "阶段二_异常修复模型.pt", map_location="cpu", weights_only=False
    )
    repair.load_state_dict(repair_checkpoint["模型参数"])
    repair.eval()
    segmentation = 轻量裂缝UNet(in_channels=4)
    segmentation_checkpoint = torch.load(
        配置.阶段二目录 / "阶段二_修复引导分割模型.pt",
        map_location="cpu",
        weights_only=False,
    )
    segmentation.load_state_dict(segmentation_checkpoint["模型参数"])
    return repair, segmentation


def 生成真实批次(generator: 在线合成批次, batch_size: int) -> Tensor:
    """只从原图裁取无标签图块，不进行裂缝合成。"""

    patches: list[np.ndarray] = []
    for _ in range(batch_size):
        patch, _, _, _ = generator.获取图块()
        patches.append(patch.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(patches))


def 弱增强(images: Tensor) -> Tensor:
    """Teacher使用的温和光度扰动，不改变裂缝几何位置。"""

    batch = images.shape[0]
    brightness = torch.empty(batch, 1, 1, 1).uniform_(-0.025, 0.025)
    contrast = torch.empty(batch, 1, 1, 1).uniform_(0.96, 1.04)
    noise = torch.randn_like(images) * 0.004
    return torch.clamp(images * contrast + brightness + noise, 0.0, 1.0)


def 强增强(images: Tensor) -> Tensor:
    """Student使用的较强光度、噪声和局部遮挡扰动。"""

    batch, _, height, width = images.shape
    brightness = torch.empty(batch, 1, 1, 1).uniform_(-0.07, 0.07)
    contrast = torch.empty(batch, 1, 1, 1).uniform_(0.86, 1.14)
    result = torch.clamp(images * contrast + brightness + torch.randn_like(images) * 0.018, 0.0, 1.0)
    for index in range(batch):
        if random.random() < 0.45:
            box_h = random.randint(10, 28)
            box_w = random.randint(8, 24)
            y0 = random.randint(0, height - box_h)
            x0 = random.randint(0, width - box_w)
            result[index, :, y0 : y0 + box_h, x0 : x0 + box_w] = images[index].mean()
    return result


def 生成伪标签(
    probabilities: Tensor,
    residual: Tensor,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """结合Teacher置信度、熵和修复残差生成正、负、忽略三状态伪标签。"""

    flat_prob = probabilities.detach().flatten()
    flat_residual = residual.detach().flatten()
    positive_threshold = max(0.72, float(torch.quantile(flat_prob, 0.992).item()))
    negative_threshold = min(0.18, float(torch.quantile(flat_prob, 0.25).item()))
    residual_threshold = float(torch.quantile(flat_residual, 0.72).item())
    entropy = -probabilities * torch.log(probabilities + 1.0e-6) - (
        1.0 - probabilities
    ) * torch.log(1.0 - probabilities + 1.0e-6)
    positive = (probabilities >= positive_threshold) & (residual >= residual_threshold) & (entropy < 0.58)
    negative = (probabilities <= negative_threshold) & (entropy < 0.48)
    valid = positive | negative
    labels = positive.float()
    total = float(probabilities.numel())
    stats = {
        "正伪标签比例": float(positive.sum().item()) / total,
        "负伪标签比例": float(negative.sum().item()) / total,
        "忽略比例": 1.0 - float(valid.sum().item()) / total,
        "平均预测熵": float(entropy.mean().item()),
        "正类概率阈值": positive_threshold,
        "负类概率阈值": negative_threshold,
        "残差阈值": residual_threshold,
    }
    return labels, valid.float(), stats


def 伪标签损失(logits: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    """仅在可信区域计算加权BCE，避免大量背景淹没少量正伪标签。"""

    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    class_weight = 1.0 + 4.0 * labels
    return (loss * class_weight * valid).sum() / (valid.sum() + 1.0)


def EMA更新(teacher: nn.Module, student: nn.Module, alpha: float) -> None:
    """使用Student参数的指数滑动平均更新Teacher，不参与反向传播。"""

    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(alpha).add_(student_parameter, alpha=1.0 - alpha)


def 绘制自训练曲线(rows: Sequence[dict[str, float | int]], path: Path) -> None:
    """上图展示三类损失，下图展示伪标签覆盖率与平均熵。"""

    x = [int(row["迭代次数"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.0), dpi=150, sharex=True)
    axes[0].plot(x, [float(row["总损失"]) for row in rows], color="#2F6B9A", label="总损失")
    axes[0].plot(x, [float(row["合成监督损失"]) for row in rows], color="#D39C2C", label="合成监督损失")
    axes[0].plot(x, [float(row["真实伪标签损失"]) for row in rows], color="#6B7280", label="真实伪标签损失")
    axes[0].set_ylabel("损失")
    axes[0].set_title("阶段三Teacher-Student自训练损失")
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(x, [float(row["正伪标签比例"]) for row in rows], color="#2F6B9A", label="正伪标签比例")
    axes[1].plot(x, [float(row["负伪标签比例"]) for row in rows], color="#D39C2C", label="负伪标签比例")
    axes[1].plot(x, [float(row["平均预测熵"]) for row in rows], color="#6B7280", linestyle="--", label="平均预测熵")
    axes[1].set_xlabel("迭代次数")
    axes[1].set_ylabel("比例或熵")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axes[1].legend(frameon=False, ncol=3)
    fig.text(0.5, 0.01, "真实图像无人工标签；中等置信像素不参与伪标签损失", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制三阶段指标对比(stage1: dict[str, float], stage2: dict[str, float], stage3: dict[str, float], path: Path) -> None:
    """比较三阶段在相同定义合成验证集上的核心指标。"""

    names = ["精确率", "召回率", "F1", "IoU"]
    keys = ["精确率Precision", "召回率Recall", "F1分数", "交并比IoU"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 5.1), dpi=150)
    series = [
        ("阶段一合成监督", stage1, "#AFC8DA", "#2F6B9A", -width),
        ("阶段二修复引导", stage2, "#E5C273", "#8A6415", 0.0),
        ("阶段三EMA Teacher", stage3, "#9AA2AA", "#4B5563", width),
    ]
    for label, metrics, color, edge, offset in series:
        bars = ax.bar(x + offset, [metrics[key] for key in keys], width, label=label, color=color, edgecolor=edge)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("三个阶段的合成验证指标对比")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, ncol=3)
    fig.text(0.5, 0.01, "合成验证用于模型消融；真实F1必须由阶段四新审核标签计算", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 滑窗预测整图(
    repair_model: 轻量修复UNet,
    segmentation_model: 轻量裂缝UNet,
    image: np.ndarray,
) -> np.ndarray:
    """对完整360°展开图进行周期滑窗概率预测并融合重叠区域。"""

    size = 配置.图块尺寸
    stride = size // 2
    height, width = image.shape[:2]
    y_starts = list(range(0, max(1, height - size + 1), stride))
    if not y_starts or y_starts[-1] != height - size:
        y_starts.append(height - size)
    x_starts = list(range(0, width, stride))
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    repair_model.eval()
    segmentation_model.eval()
    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                patch = 循环裁剪图块(image, x0, y0, size)
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)[None])
                repaired = repair_model(tensor)
                probability = torch.sigmoid(
                    segmentation_model(构造修复引导输入(tensor, repaired))
                )[0, 0].cpu().numpy()
                for local_x in range(size):
                    global_x = (x0 + local_x) % width
                    accumulator[y0 : y0 + size, global_x] += probability[:, local_x]
                    weights[y0 : y0 + size, global_x] += 1.0
    probability = accumulator / np.maximum(weights, 1.0)
    probability[: 配置.有效区顶部] = 0.0
    return np.clip(probability, 0.0, 1.0)


def 概率着色(probability: np.ndarray) -> np.ndarray:
    """将裂缝概率映射为黑—橙—白热力图。"""

    return 残差着色(probability / 6.0)


def 保存最终预测对比(original: np.ndarray, probability: np.ndarray, path: Path) -> None:
    """保存原图、概率、二值Mask和红色叠加四联对比图。"""

    binary = probability >= 配置.二值阈值
    mask_image = np.where(binary, 0, 255).astype(np.uint8)
    overlay = original.copy()
    overlay[binary] = 0.55 * overlay[binary] + 0.45 * np.array([1.0, 0.0, 0.0])
    mask_rgb = np.repeat(mask_image[..., None], 3, axis=2).astype(np.float32) / 255.0
    panels = [转为PIL(original), 转为PIL(概率着色(probability)), 转为PIL(mask_rgb), 转为PIL(overlay)]
    labels = ["原始钻孔图像", "Teacher裂缝概率", "二值裂缝Mask", "原图裂缝叠加"]
    panel_width, panel_height, gap, label_height = 244, 1350, 20, 58
    canvas = Image.new("RGB", (4 * panel_width + 5 * gap, panel_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = 加载中文字体(17)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = gap + index * (panel_width + gap)
        canvas.paste(panel, (left, 0))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((left + (panel_width - (box[2] - box[0])) // 2, panel_height + 14), label, font=font, fill="#18222B")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def 运行阶段三() -> None:
    """在真实无标签图像上执行合成监督约束下的EMA Teacher-Student自训练。"""

    设置随机种子(配置.随机种子 + 300)
    paths = 获取输入图像()
    train_paths, validation_paths = paths[:8], paths[8:]
    output = 配置.阶段三目录
    output.mkdir(parents=True, exist_ok=True)
    repair_model, student = 加载阶段二模型()
    teacher = 轻量裂缝UNet(in_channels=4)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for parameter in repair_model.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=2.5e-4, weight_decay=1.0e-4)
    real_generator = 在线合成批次(train_paths, 配置.随机种子 + 311)
    synthetic_generator = 在线合成批次(train_paths, 配置.随机种子 + 313)
    history: list[dict[str, float | int]] = []

    for iteration in range(1, 配置.阶段三迭代次数 + 1):
        real_images = 生成真实批次(real_generator, 配置.批大小)
        weak = 弱增强(real_images)
        strong = 强增强(real_images)
        with torch.no_grad():
            weak_repaired = repair_model(weak)
            teacher_probabilities = torch.sigmoid(
                teacher(构造修复引导输入(weak, weak_repaired))
            )
            residual = torch.mean(torch.abs(weak - weak_repaired), dim=1, keepdim=True)
            pseudo_labels, valid_mask, pseudo_stats = 生成伪标签(teacher_probabilities, residual)
            strong_repaired = repair_model(strong)

        synthetic_images, _, synthetic_masks = synthetic_generator.生成修复批次(配置.批大小)
        with torch.no_grad():
            synthetic_repaired = repair_model(synthetic_images)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        supervised = 分割损失(
            student(构造修复引导输入(synthetic_images, synthetic_repaired)),
            synthetic_masks,
        )
        pseudo = 伪标签损失(
            student(构造修复引导输入(strong, strong_repaired)),
            pseudo_labels,
            valid_mask,
        )
        ramp = min(1.0, iteration / max(1.0, 配置.阶段三迭代次数 * 0.35))
        total = supervised + 0.35 * ramp * pseudo
        total.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        EMA更新(teacher, student, 配置.Teacher_EMA系数)
        history.append(
            {
                "迭代次数": iteration,
                "总损失": float(total.item()),
                "合成监督损失": float(supervised.item()),
                "真实伪标签损失": float(pseudo.item()),
                **pseudo_stats,
            }
        )
        if iteration == 1 or iteration % 20 == 0:
            print(
                f"阶段三迭代 {iteration:03d}/{配置.阶段三迭代次数}，"
                f"总损失={total.item():.6f}，正伪标签={pseudo_stats['正伪标签比例']:.3%}"
            )

    torch.save({"模型参数": student.state_dict(), "输入通道": 4}, output / "阶段三_Student模型.pt")
    torch.save(
        {"模型参数": teacher.state_dict(), "输入通道": 4, "EMA系数": 配置.Teacher_EMA系数},
        output / "阶段三_EMA-Teacher模型.pt",
    )
    counts, metrics = 验证修复引导分割器(
        repair_model,
        teacher,
        在线合成批次(validation_paths, 配置.随机种子 + 229),
        配置.阶段一验证样本数,
    )
    stage1 = json.loads((配置.阶段一目录 / "阶段一_合成验证指标.json").read_text(encoding="utf-8"))["指标"]
    stage2_report = json.loads((配置.阶段二目录 / "阶段二_异常修复与分割指标.json").read_text(encoding="utf-8"))
    stage2 = stage2_report["阶段二修复引导分割指标"]
    report = {
        "阶段": "阶段三_Teacher-Student自训练",
        "说明": "真实图像只用于无标签一致性训练；下列F1仍为合成验证F1，不是独立真实审核F1。",
        "EMA系数": 配置.Teacher_EMA系数,
        "最终伪标签统计": {key: value for key, value in history[-1].items() if key not in {"迭代次数", "总损失", "合成监督损失", "真实伪标签损失"}},
        "混淆统计": counts,
        "合成验证指标": metrics,
    }
    保存JSON(report, output / "阶段三_Teacher-Student训练指标.json")
    with (output / "阶段三_自训练逐迭代数据.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    绘制自训练曲线(history, output / "阶段三_迭代次数损失与伪标签曲线.png")
    绘制混淆矩阵(counts, output / "阶段三_指数移动平均教师模型合成验证混淆矩阵.png")
    绘制三阶段指标对比(stage1, stage2, metrics, output / "阶段一至阶段三指标对比图.png")

    probability_dir = output / "原图裂缝概率图"
    mask_dir = output / "原图二值裂缝掩膜"
    overlay_dir = output / "原图裂缝叠加图"
    comparison_dir = output / "原图识别四联对比图"
    for directory in (probability_dir, mask_dir, overlay_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        original = 读取RGB(path)
        probability = 滑窗预测整图(repair_model, teacher, original)
        binary = probability >= 配置.二值阈值
        mask = np.where(binary, 0, 255).astype(np.uint8)
        overlay = original.copy()
        overlay[binary] = 0.55 * overlay[binary] + 0.45 * np.array([1.0, 0.0, 0.0])
        转为PIL(概率着色(probability)).save(probability_dir / f"{path.stem}_裂缝概率图.png")
        Image.fromarray(mask, mode="L").save(mask_dir / f"{path.stem}_二值裂缝掩膜.png")
        转为PIL(overlay).save(overlay_dir / f"{path.stem}_裂缝叠加图.png")
        保存最终预测对比(original, probability, comparison_dir / f"{path.stem}_裂缝识别四联对比图.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def 加载阶段三Teacher() -> tuple[轻量修复UNet, 轻量裂缝UNet]:
    """加载阶段三EMA Teacher，保持阶段四与已输出预测一致。"""

    repair, _ = 加载阶段二模型()
    teacher = 轻量裂缝UNet(in_channels=4)
    checkpoint = torch.load(
        配置.阶段三目录 / "阶段三_EMA-Teacher模型.pt",
        map_location="cpu",
        weights_only=False,
    )
    teacher.load_state_dict(checkpoint["模型参数"])
    teacher.eval()
    return repair, teacher


def 文件SHA256(path: Path) -> str:
    """计算文件指纹，用于证明原始JPG未被流水线覆盖。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 选择审核窗口(probability: np.ndarray) -> list[dict[str, float | int | str]]:
    """
    对独立留出图像执行与模型输出无关的纵向系统抽样。

    抽样位置不依赖预测概率，防止只挑选“模型擅长”或“高响应”区域
    造成真实F1乐观偏差；概率与熵仅写入清单供误差分层分析。
    """

    height = probability.shape[0]
    window_height = 配置.真实审核窗口高度
    last = height - window_height
    starts = np.linspace(配置.有效区顶部, last, num=8).round().astype(int).tolist()
    candidates: list[dict[str, float | int | str]] = []
    for index, y0 in enumerate(starts, start=1):
        crop = probability[y0 : y0 + window_height]
        flat = crop.reshape(-1)
        top_count = max(1, int(flat.size * 0.05))
        high_score = float(np.partition(flat, -top_count)[-top_count:].mean())
        clipped = np.clip(flat, 1.0e-6, 1.0 - 1.0e-6)
        entropy = float(np.mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))))
        candidates.append(
            {
                "纵向起点": y0,
                "高响应得分": high_score,
                "平均概率": float(flat.mean()),
                "平均熵": entropy,
                "审核类型": f"留出集系统抽样窗口{index:02d}",
            }
        )
    return candidates


def 保存审核参考四联图(
    original: np.ndarray, probability: np.ndarray, path: Path
) -> None:
    """保存留出窗口的模型概率、掩膜和叠加图，供自动审核追溯。"""

    binary = probability >= 配置.二值阈值
    mask = np.repeat(np.where(binary, 0, 255)[..., None], 3, axis=2).astype(np.float32) / 255.0
    overlay = original.copy()
    overlay[binary] = 0.55 * overlay[binary] + 0.45 * np.array([1.0, 0.0, 0.0])
    panels = [original, 概率着色(probability), mask, overlay]
    labels = ["原始留出窗口", "教师模型概率", "固定阈值模型掩膜", "模型叠加结果"]
    panel_width = original.shape[1]
    panel_height = original.shape[0]
    gap, label_height = 16, 46
    canvas = Image.new("RGB", (4 * panel_width + 5 * gap, panel_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = 加载中文字体(15)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = gap + index * (panel_width + gap)
        canvas.paste(转为PIL(panel), (left, label_height))
        draw.text((left, 13), label, fill="#203A4F", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def 绘制无标签审核诊断(rows: Sequence[dict[str, object]], path: Path) -> None:
    """将真实留出窗口的预测分布与稳定性分开展示。"""

    x = np.arange(1, len(rows) + 1)
    positive = [float(row["预测裂缝像素比例"]) for row in rows]
    uncertain = [float(row["不确定像素比例"]) for row in rows]
    light_mae = [float(row["光度扰动概率差平均值"]) for row in rows]
    shift_mae = [float(row["周期平移概率差平均值"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), dpi=150, sharex=True)
    axes[0].plot(x, positive, color="#2F6B9A", marker="o", linewidth=1.8, label="预测裂缝像素比例")
    axes[0].plot(x, uncertain, color="#D69E2E", marker="s", linewidth=1.8, label="不确定像素比例")
    axes[0].set_ylabel("像素比例")
    axes[0].set_ylim(0.0, max(0.10, max(positive + uncertain) * 1.15))
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, light_mae, color="#52606D", marker="o", linewidth=1.8, label="光度扰动一致性误差")
    axes[1].plot(x, shift_mae, color="#7C9EB2", marker="s", linewidth=1.8, label="周期平移一致性误差")
    axes[1].set_ylabel("概率平均绝对差")
    axes[1].set_xlabel("独立留出窗口序号")
    axes[1].set_xticks(x)
    axes[1].grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("独立真实留出集无标签自动审核")
    fig.text(0.5, 0.01, "样本=图1-9和图1-10的16个系统抽样窗口；该图是稳定性诊断，不是真实精度", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制无标签审核汇总(summary: dict[str, float], path: Path) -> None:
    """以单位一致的比例指标绘制自动审核汇总图。"""

    names = ["裂缝预测比例", "不确定像素比例", "高置信像素比例", "双阈值稳定度"]
    keys = ["平均预测裂缝像素比例", "平均不确定像素比例", "平均高置信像素比例", "平均双阈值稳定度"]
    values = [summary[key] for key in keys]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
    bars = ax.bar(names, values, color="#2F6B9A", edgecolor="#203A4F")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("比例或交并比")
    ax.set_title("真实留出集自动审核汇总")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.text(0.5, 0.01, "无人工真值；指标表示模型响应与稳定性，不代表F1或IoU精度", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 绘制合成验证趋势(阶段指标: list[dict[str, float]], path: Path) -> None:
    """单独展示前三阶段的合成验证趋势，不与真实审核数据混画。"""

    names = ["合成监督", "修复引导", "Teacher"]
    series = [
        ("精确率", "精确率Precision", "#2F6B9A"),
        ("召回率", "召回率Recall", "#D69E2E"),
        ("F1", "F1分数", "#52606D"),
        ("IoU", "交并比IoU", "#7C9EB2"),
    ]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    for label, key, color in series:
        ax.plot(x, [stage[key] for stage in 阶段指标], marker="o", linewidth=2, label=label, color=color)
    ax.set_xticks(x, labels=names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("前三阶段合成验证指标趋势")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, ncol=4)
    fig.text(0.5, 0.01, "数据集=在线合成裂缝验证集；真实留出集只做无标签稳定性审核", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def 运行阶段四() -> None:
    """
    对独立真实留出图像执行完全自动的无标签稳定性审核。

    本阶段不需要人工标注。无真值时不计算真实F1，也不将稳定性指标
    伪装成分割精度。
    """

    设置随机种子(配置.随机种子)
    paths = 获取输入图像()
    audit_paths = paths[8:]
    output = 配置.阶段四目录
    # 阶段四产物全部由程序生成，重跑时清除旧版本的人工标注入口。
    if output.exists():
        shutil.rmtree(output)
    raw_dir = output / "独立留出原图窗口"
    reference_dir = output / "模型自动诊断四联图"
    probability_dir = output / "模型概率数值图"
    final_mask_dir = output / "最终二值裂缝图"
    final_comparison_dir = output / "最终裂缝识别对比图"
    for directory in (
        raw_dir,
        reference_dir,
        probability_dir,
        final_mask_dir,
        final_comparison_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    repair, teacher = 加载阶段三Teacher()
    audit_records: list[dict[str, object]] = []
    audit_index = 1
    for image_path in audit_paths:
        original = 读取RGB(image_path)
        probability = 滑窗预测整图(repair, teacher, original)
        # 温和光度扰动不改变几何位置，用于检查预测光度稳定性。
        light_image = np.clip(original * 1.04 + 0.01, 0.0, 1.0)
        light_probability = 滑窗预测整图(repair, teacher, light_image)
        # 横向周期平移后再反向对齐，检查360°展开边界的等变性。
        shift = original.shape[1] // 4
        shifted_image = np.roll(original, shift, axis=1)
        shifted_probability = np.roll(
            滑窗预测整图(repair, teacher, shifted_image), -shift, axis=1
        )
        repaired = 滑窗修复整图(repair, original)
        residual = np.mean(np.abs(original - repaired), axis=2)
        periodic_seam_difference = float(np.mean(np.abs(probability[:, 0] - probability[:, -1])))
        for candidate in 选择审核窗口(probability):
            y0 = int(candidate["纵向起点"])
            y1 = y0 + 配置.真实审核窗口高度
            category = str(candidate["审核类型"])
            audit_id = f"审核{audit_index:03d}_{image_path.stem}_{category}_纵向{y0:04d}-{y1 - 1:04d}"
            raw_crop = original[y0:y1]
            probability_crop = probability[y0:y1]
            light_crop = light_probability[y0:y1]
            shifted_crop = shifted_probability[y0:y1]
            residual_crop = residual[y0:y1]
            clipped = np.clip(probability_crop, 1.0e-6, 1.0 - 1.0e-6)
            entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
            low_threshold_mask = probability_crop >= 0.45
            high_threshold_mask = probability_crop >= 0.55
            union = int(np.logical_or(low_threshold_mask, high_threshold_mask).sum())
            threshold_stability = (
                float(np.logical_and(low_threshold_mask, high_threshold_mask).sum() / union)
                if union > 0
                else 1.0
            )
            flat_probability = probability_crop.reshape(-1)
            flat_residual = residual_crop.reshape(-1)
            residual_correlation = (
                float(np.corrcoef(flat_probability, flat_residual)[0, 1])
                if np.std(flat_probability) > 1.0e-8 and np.std(flat_residual) > 1.0e-8
                else 0.0
            )
            转为PIL(raw_crop).save(raw_dir / f"{audit_id}_独立留出原图.png")
            Image.fromarray(np.round(probability_crop * 255.0).astype(np.uint8), mode="L").save(
                probability_dir / f"{audit_id}_模型概率数值图.png"
            )
            保存审核参考四联图(
                raw_crop,
                probability_crop,
                reference_dir / f"{audit_id}_模型自动诊断四联图.png",
            )
            audit_records.append(
                {
                    "审核ID": audit_id,
                    "原始图像": image_path.name,
                    "审核类型": category,
                    "纵向起点_含": y0,
                    "纵向终点_不含": y1,
                    "窗口宽": original.shape[1],
                    "窗口高": raw_crop.shape[0],
                    "模型平均概率": float(candidate["平均概率"]),
                    "模型平均熵": float(candidate["平均熵"]),
                    "预测裂缝像素比例": float(np.mean(probability_crop >= 配置.二值阈值)),
                    "不确定像素比例": float(np.mean((probability_crop >= 0.35) & (probability_crop <= 0.65))),
                    "高置信像素比例": float(np.mean((probability_crop <= 0.10) | (probability_crop >= 0.90))),
                    "平均像素熵": float(entropy.mean()),
                    "光度扰动概率差平均值": float(np.mean(np.abs(probability_crop - light_crop))),
                    "周期平移概率差平均值": float(np.mean(np.abs(probability_crop - shifted_crop))),
                    "双阈值稳定度": threshold_stability,
                    "修复残差平均值": float(residual_crop.mean()),
                    "预测概率与修复残差相关系数": residual_correlation,
                    "整图周期边界概率差": periodic_seam_difference,
                }
            )
            audit_index += 1

    manifest_path = output / "独立真实无标签审核逐窗口数据.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_records[0]))
        writer.writeheader()
        writer.writerows(audit_records)

    汇总 = {
        "平均预测裂缝像素比例": float(np.mean([row["预测裂缝像素比例"] for row in audit_records])),
        "平均不确定像素比例": float(np.mean([row["不确定像素比例"] for row in audit_records])),
        "平均高置信像素比例": float(np.mean([row["高置信像素比例"] for row in audit_records])),
        "平均像素熵": float(np.mean([row["平均像素熵"] for row in audit_records])),
        "平均光度扰动概率差": float(np.mean([row["光度扰动概率差平均值"] for row in audit_records])),
        "平均周期平移概率差": float(np.mean([row["周期平移概率差平均值"] for row in audit_records])),
        "平均双阈值稳定度": float(np.mean([row["双阈值稳定度"] for row in audit_records])),
        "平均修复残差": float(np.mean([row["修复残差平均值"] for row in audit_records])),
        "平均概率残差相关系数": float(np.mean([row["预测概率与修复残差相关系数"] for row in audit_records])),
        "平均周期边界概率差": float(np.mean([row["整图周期边界概率差"] for row in audit_records])),
    }
    绘制无标签审核诊断(audit_records, output / "独立真实留出集无标签自动审核诊断图.png")
    绘制无标签审核汇总(汇总, output / "真实留出集自动审核汇总图.png")

    stage1 = json.loads((配置.阶段一目录 / "阶段一_合成验证指标.json").read_text(encoding="utf-8"))["指标"]
    stage2 = json.loads((配置.阶段二目录 / "阶段二_异常修复与分割指标.json").read_text(encoding="utf-8"))["阶段二修复引导分割指标"]
    stage3 = json.loads((配置.阶段三目录 / "阶段三_Teacher-Student训练指标.json").read_text(encoding="utf-8"))["合成验证指标"]
    绘制合成验证趋势([stage1, stage2, stage3], output / "前三阶段合成验证指标趋势图.png")
    summary_rows: list[dict[str, object]] = []
    for stage_name, metrics in [
        ("阶段一_合成裂缝监督", stage1),
        ("阶段二_异常修复引导", stage2),
        ("阶段三_Teacher-Student", stage3),
    ]:
        summary_rows.append({"阶段": stage_name, "评价数据": "合成裂缝验证集", "状态": "已完成", **metrics})
    summary_rows.append(
        {
            "阶段": "阶段四_独立真实审核",
            "评价数据": "真实无标签留出集",
            "状态": "全自动无标签审核已完成_无真值故精度指标不可定义",
            **{key: None for key in stage1},
        }
    )
    with (output / "四阶段评价指标汇总.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    mask_validation: list[dict[str, object]] = []
    for image_path in paths:
        mask_path = 配置.阶段三目录 / "原图二值裂缝掩膜" / f"{image_path.stem}_二值裂缝掩膜.png"
        comparison_path = 配置.阶段三目录 / "原图识别四联对比图" / f"{image_path.stem}_裂缝识别四联对比图.png"
        with Image.open(mask_path) as mask_image:
            values = sorted(np.unique(np.asarray(mask_image)).tolist())
            validation_passed = mask_image.size == (244, 1350) and mask_image.mode == "L" and set(values).issubset({0, 255})
            mask_validation.append(
                {
                    "原图": image_path.name,
                    "原图SHA256": 文件SHA256(image_path),
                    "预测Mask": mask_path.name,
                    "Mask尺寸": list(mask_image.size),
                    "Mask模式": mask_image.mode,
                    "Mask像素值": values,
                    "校验通过": validation_passed,
                }
            )
        if not validation_passed:
            raise RuntimeError(f"{mask_path.name}未通过最终发布校验。")
        # 阶段三的候选预测经阶段四审核后，才复制到最终成果目录。
        shutil.copy2(mask_path, final_mask_dir / f"{image_path.stem}_最终二值裂缝图.png")
        shutil.copy2(
            comparison_path,
            final_comparison_dir / f"{image_path.stem}_最终裂缝识别对比图.png",
        )
    report = {
        "阶段": "阶段四_独立真实留出集全自动无标签审核",
        "审核策略": "仅使用未参与Teacher-Student真实无标签训练的图1-9和图1-10；每图等距抽取8个窗口，自动计算预测分布、不确定性、光度一致性、周期平移一致性、双阈值稳定性和修复残差关联",
        "Teacher-Student真实无标签训练图像": [path.name for path in paths[:8]],
        "独立留出审核图像": [path.name for path in audit_paths],
        "审核窗口数": len(audit_records),
        "人工标注需求": "不需要",
        "评价状态": "全自动无标签审核已完成",
        "二值化阈值": 配置.二值阈值,
        "独立真实无标签诊断汇总": 汇总,
        "真实F1分数": None,
        "真实混淆矩阵": None,
        "最终二值裂缝图目录": str(final_mask_dir),
        "最终裂缝识别对比图目录": str(final_comparison_dir),
        "说明": "本阶段完全自动，不需要人工标注。无真值时F1和混淆矩阵在数学上不可定义；上述结果是稳定性与不确定性诊断，不是真实精度。",
        "原图与预测Mask校验": mask_validation,
    }
    保存JSON(report, output / "阶段四_全自动无标签审核结果.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def 构建命令行() -> argparse.ArgumentParser:
    """构建分阶段命令行入口。"""

    parser = argparse.ArgumentParser(description="附件1裂缝识别四阶段流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("阶段一", help="运行合成裂缝监督阶段")
    subparsers.add_parser("阶段二", help="运行异常修复与修复引导分割阶段")
    subparsers.add_parser("阶段三", help="运行Teacher-Student真实无标签自训练阶段")
    subparsers.add_parser("阶段四", help="运行独立真实留出集全自动无标签审核")
    return parser


def main() -> None:
    args = 构建命令行().parse_args()
    if args.command == "阶段一":
        运行阶段一()
    elif args.command == "阶段二":
        运行阶段二()
    elif args.command == "阶段三":
        运行阶段三()
    elif args.command == "阶段四":
        运行阶段四()


if __name__ == "__main__":
    main()
