import torch
import clip
import numpy as np
import cv2 # SAM 需要 BGR 格式
from tqdm import tqdm
from PIL import Image

# 1. 从我们自己的文件中导入
from models import load_clip, load_sam_automask_generator, get_device
from dataloader import load_refcoco_dataset

# -----------------------------------------------------------------
# (新!) 评估辅助函数
# -----------------------------------------------------------------

def calculate_iou(mask_pred, mask_gt):
    """
    计算两个 boolean numpy 蒙版之间的 IoU (交并比)。
    
    参数:
    - mask_pred (np.array): 预测蒙版 (boolean, HxW)
    - mask_gt (np.array): 真实蒙版 (boolean, HxW)
    
    返回:
    - float: IoU 分数
    """
    if mask_pred is None or mask_gt is None:
        return 0.0
        
    # .astype(np.uint8) 是一种安全的转换方式
    intersection = np.sum((mask_pred & mask_gt).astype(np.uint8))
    union = np.sum((mask_pred | mask_gt).astype(np.uint8))
    
    if union == 0:
        # 如果并集是0 (两个蒙版都是空的), 
        # 如果交集也是0 (预测正确), IoU 是 1.0
        # 如果交集不是0 (不可能), IoU 是 0.0
        return 1.0 if intersection == 0 else 0.0
        
    iou = intersection / union
    return iou

def apply_mask_to_image(image, mask):
    """
    (从 Notebook 复制)
    将二进制蒙版应用于图像，裁剪出蒙版区域。
    返回一个 PIL Image，以便 CLIP 预处理器使用。
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
        
    # 创建一个 RGBA 图像用于裁剪
    masked_image = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    masked_image[..., :3] = image
    masked_image[mask, 3] = 255 # 只在蒙版区域设置 Alpha
    
    return Image.fromarray(masked_image, 'RGBA')

# -----------------------------------------------------------------
# 主评估流程
# -----------------------------------------------------------------

def main():
    """
    运行完整的评估流程。
    """
    
    # --- 1. 设置 ---
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 警告: 'val' split 有 8811 张图片。
    # 在单个 GPU/MPS 上运行 SAM 可能每张图需要 20-40 秒。
    # 跑完整个数据集可能需要 70+ 小时！
    #
    # 我们先将样本数限制为 20 个来进行快速测试。
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    LIMIT_SAMPLES = 20
    
    print("--- 评估脚本启动 ---")
    print(f"注意: 评估将限制在 {LIMIT_SAMPLES} 个样本上。")

    # --- 2. 加载模型 ---
    print("\n--- 正在加载模型... ---")
    # (我们使用我们修复过的 vit_b 版本)
    mask_generator, sam_device = load_sam_automask_generator()
    clip_model, clip_preprocess, clip_device = load_clip()

    if mask_generator is None or clip_model is None:
        print("模型加载失败。退出。")
        return

    # --- 3. 加载数据集 ---
    print("\n--- 正在加载 RefCOCO 'val' split... ---")
    # (我们使用我们修复过的 Hugging Face Dataloader)
    dataset = load_refcoco_dataset(split_name='val')
    
    if not dataset:
        print("数据加载失败。退出。")
        return

    # --- 4. 运行评估循环 ---
    print(f"\n--- 开始评估 {LIMIT_SAMPLES} 个样本... ---")
    
    iou_scores = []
    
    # 使用 tqdm 创建一个进度条
    pbar = tqdm(dataset[:LIMIT_SAMPLES], desc="正在评估")
    
    for item in pbar:
        try:
            image_rgb = item['image_rgb']
            text_prompt = item['text_prompt']
            gt_mask = item['ground_truth_mask']
            
            # a. 编码文本 (一次)
            with torch.no_grad():
                text = clip.tokenize([text_prompt]).to(clip_device)
                text_features = clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # b. SAM 生成所有蒙版
            # (SAM 需要 BGR 格式)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            sam_masks = mask_generator.generate(image_bgr)

            if not sam_masks:
                # SAM 找不到任何东西。这个样本的 IoU 是 0
                iou_scores.append(0.0)
                continue

            # c. CLIP 排名 (内存安全版)
            best_score = -1.0
            best_mask = None

            for ann in sam_masks:
                mask = ann['segmentation'] # (H, W) boolean 数组
                
                with torch.no_grad():
                    # i. 裁剪
                    cropped_pil_image = apply_mask_to_image(image_rgb, mask)
                    
                    # ii. 创建 batch_size = 1
                    image_input = clip_preprocess(cropped_pil_image).unsqueeze(0).to(clip_device)
                    
                    # iii. 编码图像
                    image_features = clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # iv. 评分 (包含 MPS .float() 修复)
                    # 我们只需要原始的 "logit" 分数，不需要 softmax
                    score = (100.0 * text_features.float() @ image_features.float().T).item()

                    # v. 查找最佳
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                
                # (清理内存)
                del image_input, image_features, cropped_pil_image, mask

            # d. 计算 IoU
            # 我们现在有了 `best_mask` (预测) 和 `gt_mask` (答案)
            iou = calculate_iou(best_mask, gt_mask)
            iou_scores.append(iou)
            
            # 更新进度条的描述
            pbar.set_description(f"当前 mIoU: {np.mean(iou_scores):.4f}")

        except Exception as e:
            print(f"处理样本 {item.get('image_id')} 时发生意外错误: {e}")
            iou_scores.append(0.0) # 记为失败

    # --- 5. 报告结果 ---
    print("\n--- 评估完成 ---")
    
    if iou_scores:
        final_miou = np.mean(iou_scores)
        print(f"\n在 {len(iou_scores)} 个样本上的最终 mIoU: {final_miou:.4f}")
    else:
        print("未能计算任何分数。")

if __name__ == "__main__":
    main()

