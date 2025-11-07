import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import sys
from PIL import Image, ImageDraw # 导入 ImageDraw 来绘制蒙版

# -----------------------------------------------------------------
# (新!) 辅助函数：从多边形坐标创建蒙版
# -----------------------------------------------------------------
def create_mask_from_polygon(segmentation_coords, image_size):
    """
    根据 COCO 格式的多边形坐标列表创建一个 boolean 蒙版。

    参数:
    - segmentation_coords (list): 坐标列表, e.g., [x1, y1, x2, y2, ...]
    - image_size (tuple): (width, height) 图像的尺寸
    
    返回:
    - np.array (boolean): (height, width) 的蒙版数组
    """
    # 创建一个全黑的 'L' (8-bit grayscale) 图像
    mask_img = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask_img)
    
    # 在蒙版上用白色 (1) 绘制多边形
    draw.polygon(segmentation_coords, outline=1, fill=1)
    
    # 将 PIL 图像转换为 boolean numpy 数组
    mask = np.array(mask_img) > 0
    return mask

# -----------------------------------------------------------------
# 主加载函数 (已更新)
# -----------------------------------------------------------------
def load_refcoco_dataset(split_name='val'):
    """
    使用 Hugging Face `datasets` 库加载 RefCOCO "考题列表"。
    (已更新，以匹配 'lmms-lab/RefCOCO' 的真实数据结构)
    """
    print(f"--- 正在使用 Hugging Face 加载 'lmms-lab/RefCOCO' (Split: {split_name}) ---")
    
    try:
        ds = load_dataset("lmms-lab/RefCOCO", split=split_name)
    except Exception as e:
        print(f"从 Hugging Face 加载数据集失败: {e}")
        return []

    print(f"找到了 {len(ds)} 个样本。正在转换格式并清理数据...")

    dataset_items = []
    
    # (新!) 定义“好”样本必须拥有的 *真实* 键
    required_keys = ['image', 'segmentation', 'answer', 'file_name']

    for item in tqdm(ds, desc=f"处理 {split_name}"):
        
        # 1. 清理逻辑 (使用正确的键)
        # 确保键存在，且 'answer' 和 'segmentation' 不是空的
        if (not all(key in item and item[key] is not None for key in required_keys) 
            or not item['answer'] or not item['segmentation']):
            continue

        try:
            # 2. (新!) 转换数据
            
            # a. 图像: 从 PIL Image 转换为 Numpy RGB
            image_pil = item['image'].convert('RGB')
            image_rgb = np.array(image_pil)
            
            # b. (新!) 蒙版: 从多边形坐标 (segmentation) 绘制蒙版
            gt_mask = create_mask_from_polygon(item['segmentation'], image_pil.size)
            
            # c. (新!) 文本提示: 我们取 'answer' 列表中的第一个答案
            text_prompt = item['answer'][0]
            
            # d. (新!) 图像 ID
            image_id = item['file_name']
            
            # 3. 打包成一个"考题"
            dataset_items.append({
                'image_id': image_id,
                'image_rgb': image_rgb,
                'text_prompt': text_prompt,
                'ground_truth_mask': gt_mask
            })
        except Exception as e:
            # 捕获其他意外错误 (例如图像转换失败)
            print(f"\n处理样本 {item.get('file_name')} 时出错 (意外): {e}")
            continue

    print(f"--- Dataloader 加载完毕。成功加载 {len(dataset_items)} / {len(ds)} 个样本 ---")
    return dataset_items

# -----------------------------------------------------------------
# 主测试 (当你直接运行 `python src/data.py` 时)
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("--- 正在测试 RefCOCO Dataloader (Hugging Face 版本) ---")
    
    # 我们加载 'val' split 来进行测试
    test_dataset = load_refcoco_dataset(split_name='val')
    
    if test_dataset:
        print(f"\n--- 测试成功 ---")
        print(f"总共加载了 {len(test_dataset)} 个样本。")
        
        # 打印第一个样本的信息
        print("\n第一个样本 (Example):")
        sample = test_dataset[0]
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Text Prompt: \"{sample['text_prompt']}\"")
        print(f"  Image Shape: {sample['image_rgb'].shape}")
        print(f"  GT Mask Shape: {sample['ground_truth_mask'].shape}")
        print(f"  GT Mask Type: {sample['ground_truth_mask'].dtype}")
        
        # (新!) 验证蒙版是否正确加载
        print(f"  蒙版中 'True' 的像素点数量: {np.sum(sample['ground_truth_mask'])}")

    else:
        print("\n--- 测试失败 ---")
        print("未能加载任何数据。")



    

