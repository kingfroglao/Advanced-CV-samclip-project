import torch
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os
from urllib.request import urlretrieve
import warnings
import sys
import numpy as np
from PIL import Image, ImageDraw

# --- (新!) 从 transformers 导入 pipeline ---
# 解决 "UnboundLocalError" 错误，并使用你建议的正确方法
try:
    from transformers import pipeline
except ImportError:
    print("未找到 'transformers' 库，请先安装: pip install transformers")

# -----------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------

# (vit_b)
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

def get_device():
    """
    获取可用的计算设备 (CUDA, MPS, or CPU)。
    """
    if torch.cuda.is_available():
        print("检测到 CUDA 设备。")
        return "cuda"
    if torch.backends.mps.is_available():
        print("检测到 MPS (Apple Silicon) 设备。")
        return "mps"
    print("未检测到 GPU, 使用 CPU。")
    return "cpu"

def download_sam_checkpoint(target_path=SAM_CHECKPOINT_PATH, url=SAM_CHECKPOINT_URL):
    if not os.path.exists(target_path):
        print(f"SAM checkpoint '{target_path}' (vit_b) 未找到，开始下载...")
        try:
            urlretrieve(url, target_path)
            print(f"SAM checkpoint (vit_b) 下载完成: {target_path}")
        except Exception as e:
            print(f"下载 SAM checkpoint 失败: {e}")
            return False
    else:
        print(f"SAM checkpoint '{target_path}' (vit_b) 已存在。")
    return True

# -----------------------------------------------------------------
# 1. 基线方法 (Baseline) 的加载器
# -----------------------------------------------------------------

def load_clip(model_name="ViT-B/32"):
    device = get_device()
    print(f"正在加载 CLIP 模型 '{model_name}' 到设备: {device}...")
    try:
        model, preprocess = clip.load(model_name, device=device)
        print("CLIP 模型加载成功。")
        return model, preprocess, device
    except Exception as e:
        print(f"加载 CLIP 模型失败: {e}")
        return None, None, None

def load_sam_automask_generator(model_type=SAM_MODEL_TYPE, checkpoint_path=SAM_CHECKPOINT_PATH):
    device = get_device()
    print(f"正在加载 SAM Automatic Mask Generator '{model_type}' 到设备: {device}...")
    
    if not download_sam_checkpoint():
        print("无法加载 SAM，因为 checkpoint 不可用。")
        return None
        
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # (MPS 修复)
        if device == "mps":
            sam = sam.to(torch.float32)
            
        sam.to(device=device)
        
        # (MPS 修复) 修复 pixel_mean 和 pixel_std 仍为 float64 的问题
        if device == "mps":
            sam.pixel_mean = sam.pixel_mean.to(torch.float32)
            sam.pixel_std = sam.pixel_std.to(torch.float32)

        mask_generator = SamAutomaticMaskGenerator(model=sam)
        print(f"SAM Automatic Mask Generator ({model_type}) 加载成功。")
        return mask_generator, device
    except Exception as e:
        print(f"加载 SAM Automatic Mask Generator 失败: {e}")
        return None, None

# -----------------------------------------------------------------
# 2. 增强方法 (Method+) 的加载器
# -----------------------------------------------------------------

def load_grounding_dino_pipeline(model_name="IDEA-Research/grounding-dino-base"):
    """
    (新!) 使用你建议的 transformers pipeline API 加载 Grounding DINO。
    """
    device = get_device()
    print(f"正在从 Hugging Face 加载 Grounding DINO pipeline '{model_name}'...")
    
    try:
        # 将 "cuda" 或 "mps" 映射到设备 ID 0, CPU 映射到 -1
        # 这是 pipeline API 的要求
        torch_device = 0 if device in ["cuda", "mps"] else -1
        
        gd_pipeline = pipeline(
            "zero-shot-object-detection",
            model=model_name,
            device=torch_device # 传递设备 ID
        )
        print("Grounding DINO pipeline 加载成功。")
        return gd_pipeline, device
    except Exception as e:
        print(f"加载 Grounding DINO pipeline 失败: {e}")
        return None, None

def load_sam_predictor(model_type=SAM_MODEL_TYPE, checkpoint_path=SAM_CHECKPOINT_PATH):
    """
    加载 SAM Predictor (用于接收精确的提示，如边界框)。
    """
    device = get_device()
    print(f"正在加载 SAM Predictor '{model_type}' 到设备: {device}...")

    if not download_sam_checkpoint():
        print("无法加载 SAM，因为 checkpoint 不可用。")
        return None
        
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        if device == "mps":
            sam = sam.to(torch.float32)
            
        sam.to(device=device)

        if device == "mps":
            sam.pixel_mean = sam.pixel_mean.to(torch.float32)
            sam.pixel_std = sam.pixel_std.to(torch.float32)
            
        predictor = SamPredictor(sam)
        print(f"SAM Predictor ({model_type}) 加载成功。")
        return predictor, device
    except Exception as e:
        print(f"加载 SAM Predictor 失败: {e}")
        return None, None

# -----------------------------------------------------------------
# 主测试 (当你直接运行 `python modelpart2.py` 时)
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("--- 正在测试所有模型加载器 ---")

    print("\n--- 1. 测试基线方法加载 ---")
    load_clip()
    load_sam_automask_generator()

    print("\n--- 2. 测试增强方法加载 ---")
    load_grounding_dino_pipeline() # (新!) 测试你的 pipeline 加载器
    load_sam_predictor()
    
    print("\n--- 所有模型加载器测试完毕 ---")
    print(f"你可以通过运行 'python {sys.argv[0]}' 来验证此文件。")
