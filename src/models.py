import torch
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
from urllib.request import urlretrieve
import warnings

# --- 配置 (vit_b) ---
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# --- 帮助函数 ---

def get_device():
    """
    获取可用的计算设备 (CUDA, MPS, or CPU)。
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        print("检测到 MPS (Apple Silicon) 设备。")
        return "mps"
    print("未检测到 GPU, 使用 CPU。")
    return "cpu"

def download_sam_checkpoint(target_path=SAM_CHECKPOINT_PATH, url=SAM_CHECKPOINT_URL):
    """
    如果 SAM checkpoint 文件不存在，则下载它。(vit_b)
    """
    if not os.path.exists(target_path):
        print(f"SAM checkpoint '{target_path}' (vit_b) 未找到，开始下载...")
        try:
            urlretrieve(url, target_path)
            print(f"SAM checkpoint (vit_b) 下载完成: {target_path}")
        except Exception as e:
            if "SSL: CERTIFICATE_VERIFY_FAILED" in str(e):
                print(f"下载失败: {e}")
                print("--- SSL 证书验证失败 ---")
                print("请手动从以下链接下载，并放置在项目根目录:")
                print(url)
            else:
                print(f"下载 SAM checkpoint 失败: {e}")
                print("请手动从以下链接下载，并放置在项目根目录:")
                print(url)
            return False
    else:
        print(f"SAM checkpoint '{target_path}' (vit_b) 已存在。")
    return True

# --- 模型加载器 ---

def load_clip(model_name="ViT-B/32"):
    """
    加载 CLIP 模型和图像预处理器。
    """
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
    """
    加载 SAM (Segment Anything Model) 并初始化 SamAutomaticMaskGenerator。(已更新为 vit_b)
    """
    device = get_device()
    print(f"正在加载 SAM 模型 '{model_type}' (Small Version) 到设备: {device}...")

    if not download_sam_checkpoint():
        print("无法加载 SAM 模型，因为 checkpoint 不可用。")
        return None
        
    try:
        # 1. 在 CPU 上加载模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # --- vvvv 这是 v3 版 MPS 修复 (正确的顺序) vvvv ---
        
        if device == "mps":
            print("正在应用 MPS (float32) 修复...")
            
            # 2. (关键) 先在 CPU 上把模型转为 float32
            # 这会一并转换 pixel_mean 和 pixel_std
            sam = sam.to(torch.float32) 
            
            print("MPS 修复应用完成。")
            
        # --- ^^^^ 修复结束 ^^^^ ---
            
        # 3. (最后) 才把这个 float32 的模型移动到 MPS 设备
        sam.to(device=device)
            
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print(f"SAM Automatic Mask Generator ({model_type}) 加载成功。")
        return mask_generator, device
    except Exception as e:
        print(f"加载 SAM 模型或生成器失败: {e}")
        return None, None

# --- 主测试 ---
if __name__ == "__main__":
    print("--- 测试 CLIP 加载 ---")
    clip_model, clip_preprocess, clip_device = load_clip()

    print("\n--- 测试 SAM (vit_b) 加载 ---")
    mask_gen, sam_device = load_sam_automask_generator()

