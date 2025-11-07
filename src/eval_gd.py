import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入我们的数据加载器和模型加载器
# (我们假设 data.py 和 modelpart2.py 都在同一个目录下)
import dataloader as data_loader
import modelpart2 as models

# --- 配置 ---
LIMIT_SAMPLES = 20  # (关键!) 先用 20 个样本快速测试
DATA_SPLIT = 'val'  # 使用我们加载过的 'val' split

# -----------------------------------------------------------------
# 辅助函数: 计算 IoU
# -----------------------------------------------------------------

def calculate_iou(pred_mask, gt_mask):
    """
    计算两个 boolean 蒙版 (H, W) 之间的 IoU (交并比)。
    """
    if pred_mask.shape != gt_mask.shape:
        pred_pil = Image.fromarray(pred_mask)
        gt_pil = Image.fromarray(gt_mask)
        
        pred_pil = pred_pil.resize(gt_pil.size, Image.NEAREST)
        pred_mask = np.array(pred_pil) > 0 # 转换回 boolean

    # 逻辑与 (&) 是交集
    intersection = np.logical_and(pred_mask, gt_mask)
    # 逻辑或 (|) 是并集
    union = np.logical_or(pred_mask, gt_mask)
    
    # 避免除以零
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

# -----------------------------------------------------------------
# 主评估流程
# -----------------------------------------------------------------

def run_evaluation():
    print("--- 开始评估: 增强方法 (Grounding DINO + SAM Predictor) ---")

    # --- 1. 加载模型 ---
    print("正在加载模型...")
    gd_pipeline, device = models.load_grounding_dino_pipeline()
    sam_predictor, sam_device = models.load_sam_predictor()

    if gd_pipeline is None or sam_predictor is None:
        print("!!! 致命错误: 无法加载评估所需的模型。退出。")
        return

    # --- 2. 加载数据集 (考题) ---
    print(f"正在加载 {DATA_SPLIT} split (最多 {LIMIT_SAMPLES} 个样本)...")
    dataset = data_loader.load_refcoco_dataset(split_name=DATA_SPLIT)
    
    if not dataset:
        print("!!! 致命错误: 无法加载数据集。退出。")
        return
        
    if LIMIT_SAMPLES is not None and len(dataset) > LIMIT_SAMPLES:
        dataset = dataset[:LIMIT_SAMPLES]
    
    print(f"模型和数据加载完毕。开始评估 {len(dataset)} 个样本...")

    # --- 3. 循环评估 ---
    total_iou = 0.0
    processed_samples = 0
    
    # (新!) 在循环开始前清理一次 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, sample in enumerate(tqdm(dataset, desc="评估进度")):
        
        image_rgb = sample['image_rgb']
        text_prompt = sample['text_prompt']
        gt_mask = sample['ground_truth_mask']
        image_pil = Image.fromarray(image_rgb)

        # 定义我们将在循环中创建的变量
        gd_outputs, input_box, masks, scores, logits, pred_mask = (None,)*6

        try:
            # --- 4. 增强方法的流程 ---
            
            # (a) Grounding DINO 运行
            gd_outputs = gd_pipeline(image_pil, candidate_labels=[text_prompt])
            
            # (b) 提取最佳边界框 (Box)
            if not gd_outputs:
                tqdm.write(f"\n样本 {i} ({sample['image_id']}): Grounding DINO 未找到任何对象。跳过。")
                continue
            
            best_box = gd_outputs[0]['box']
            input_box = np.array([best_box[0], best_box[1], best_box[2], best_box[3]])

            # (c) SAM Predictor 运行
            sam_predictor.set_image(image_rgb)
            masks, scores, logits = sam_predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            pred_mask = masks[0] 

            # --- 5. 比较与计分 ---
            iou = calculate_iou(pred_mask, gt_mask)
            total_iou += iou
            processed_samples += 1

        except Exception as e:
            tqdm.write(f"\n样本 {i} ({sample['image_id']}) 处理时发生意外错误: {e}")
            
        finally:
            # --- (新!) 内存清理 ---
            del gd_outputs, input_box, masks, scores, logits, pred_mask, image_pil, image_rgb, text_prompt, gt_mask
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # (调试) 实时更新 mIoU
        if processed_samples > 0 and (i + 1) % 10 == 0:
            tqdm.write(f"  [进度] 样本 {i+1}/{len(dataset)}: 当前 mIoU: {total_iou / processed_samples:.4f}")

    # --- 6. 报告最终结果 ---
    if processed_samples == 0:
        print("!!! 评估完成，但没有一个样本被成功处理。 mIoU = 0.0")
        return

    final_miou = total_iou / processed_samples
    print("\n--- 评估完成 ---")
    print(f"\n在 {processed_samples} 个样本上的最终 mIoU: {final_miou:.4f}")
    print(f"(基线 mIoU 是: 0.1104)")
    if final_miou > 0.1104:
        print("--- 成功! 增强方法的分数高于基线! ---")
    else:
        print("--- 提示: 增强方法的分数低于或等于基线。 ---")

# -----------------------------------------------------------------
# 主函数入口
# -----------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()

