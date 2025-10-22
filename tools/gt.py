

import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# 配置路径
# data_root = "/root/data1/yzj/UIIS/UDW"
# ann_file = os.path.join(data_root, "annotations", "val.json")
# img_dir = os.path.join(data_root, "val")
# output_dir = os.path.join(data_root, "UIIS_val_gt_vis")
# stats_dir = os.path.join(data_root, "stats")
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(stats_dir, exist_ok=True)

# # USIS10K配置路径
data_root = "/root/data1/yzj/USIS10K"
ann_file = os.path.join(data_root, "multi_class_annotations", "multi_class_val_annotations.json")
img_dir = os.path.join(data_root, "val")
output_dir = os.path.join(data_root, "USIS10K_val_gt_vis")
stats_dir = os.path.join(data_root, "stats")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)

# 类别配置
CLASSES = ('fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor')
PALETTE = [
    (230, 216, 173), (122, 160, 255), (144, 238, 144),
    (140, 180, 210), (193, 182, 255), (221, 160, 221), (211, 211, 211)
]
PALETTE_BGR = [tuple(reversed(c)) for c in PALETTE]  # BGR格式

# 初始化数据集级别的统计变量
dataset_stats = {
    'total_pixels': 0,
    'total_mask_pixels': 0,
    'per_class_pixels': np.zeros(len(CLASSES), dtype=np.uint64),
    'per_class_images': np.zeros(len(CLASSES), dtype=np.uint64)  # 统计每类出现的图像数
}

# 加载COCO格式的标注
coco = COCO(ann_file)
img_ids = coco.getImgIds()

# 打开文件准备记录详细统计
per_img_stats = open(os.path.join(stats_dir, "per_image_stats.csv"), "w")
per_img_stats.write("Image,Total Pixels,Mask Pixels,Mask Ratio")
for cls in CLASSES:
    per_img_stats.write(f",{cls} Pixels,{cls} Ratio")
per_img_stats.write("\n")

# 处理每张验证集图像
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    img_path = os.path.join(img_dir, img_name)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image not found at {img_path}")
        continue
    
    h, w = img.shape[:2]
    total_pixels = h * w
    mask_total = 0  # 当前图像的掩码总像素数
    
    # 为统计创建的图像级别变量
    image_stats = {
        'total_pixels': total_pixels,
        'class_pixels': np.zeros(len(CLASSES), dtype=np.uint32)
    }
    
    # 创建可视化图层
    overlay = img.copy()
    output_mask = np.zeros((h, w), dtype=np.uint8)  # 用于统计的掩码图像
    
    # 获取当前图像的所有标注
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)
    
    for ann in annotations:
        cat_id = ann['category_id'] - 1  # 类别索引从0开始
        if cat_id < 0 or cat_id >= len(CLASSES):
            continue
            
        color = PALETTE_BGR[cat_id] if cat_id < len(PALETTE_BGR) else (0, 0, 0)
        
        # 处理每个分割多边形
        mask = coco.annToMask(ann)  # 使用COCO工具生成二值掩码
        mask_area = np.sum(mask)  # 统计当前实例的像素数
        
        # 累加到统计变量
        image_stats['class_pixels'][cat_id] += mask_area
        mask_total += mask_area
        
        # 可视化填充
        for seg in ann['segmentation']:
            poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], color)
    
    # 更新数据集级统计
    dataset_stats['total_pixels'] += total_pixels
    dataset_stats['total_mask_pixels'] += mask_total
    dataset_stats['per_class_pixels'] += image_stats['class_pixels']
    for i in range(len(CLASSES)):
        if image_stats['class_pixels'][i] > 0:
            dataset_stats['per_class_images'][i] += 1
    
    # 将掩码与原始图像混合
    alpha = 0.5
    mixed = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # 在图像上添加统计信息
    mask_ratio = mask_total / total_pixels * 100
    text_y = 30
    cv2.putText(mixed, f"Mask Coverage: {mask_ratio:.2f}%", (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for i, (cls, px) in enumerate(zip(CLASSES, image_stats['class_pixels'])):
        if px > 0:
            text_y += 30
            ratio = px / total_pixels * 100
            cv2.putText(mixed, f"{cls}: {ratio:.2f}%", (10, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PALETTE_BGR[i], 2)
    
    # 添加图例说明
    legend_height = 200
    legend = np.zeros((legend_height, mixed.shape[1], 3), dtype=np.uint8) + 255
    
    class_height = legend_height // len(CLASSES)
    for i, (cls_name, color) in enumerate(zip(CLASSES, PALETTE_BGR)):
        y_start = i * class_height
        y_end = (i + 1) * class_height
        
        cv2.rectangle(legend, (10, y_start + 5), (50, y_end - 5), color, -1)
        cv2.rectangle(legend, (10, y_start + 5), (50, y_end - 5), (0, 0, 0), 1)
        cv2.putText(legend, cls_name, (60, (y_start + y_end) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 合并结果
    final = np.vstack([mixed, legend])
    
    # 保存结果
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, final)
    
    # 记录单图统计
    per_img_stats.write(f"{img_name},{total_pixels},{mask_total},{mask_ratio:.4f}")
    for px in image_stats['class_pixels']:
        ratio = px / total_pixels * 100
        per_img_stats.write(f",{px},{ratio:.4f}")
    per_img_stats.write("\n")

# 关闭文件
per_img_stats.close()

# 生成数据集级别的统计报告
with open(os.path.join(stats_dir, "dataset_summary.txt"), "w") as f:
    # 基本统计
    num_images = len(img_ids)
    f.write(f"UIIS Dataset Val Set Statistics\n")
    f.write("="*50 + "\n")
    f.write(f"Total Images: {num_images}\n")
    f.write(f"Total Pixels: {dataset_stats['total_pixels']:,}\n")
    f.write(f"Total Mask Pixels: {dataset_stats['total_mask_pixels']:,}\n")
    overall_ratio = dataset_stats['total_mask_pixels'] / dataset_stats['total_pixels'] * 100
    f.write(f"Overall Mask Coverage: {overall_ratio:.4f}%\n\n")
    
    # 类别详细统计
    f.write("Per-Class Statistics:\n")
    f.write(f"{'Class':<15} {'Pixels':<15} {'Dataset Ratio':<12} {'Img Ratio Avg':<14} {'Img Count':<10} {'Pct Images':<10}\n")
    f.write("-"*70 + "\n")
    
    class_stats = []
    for i, cls in enumerate(CLASSES):
        px = dataset_stats['per_class_pixels'][i]
        img_count = dataset_stats['per_class_images'][i]
        
        # 计算各类别占总掩码面积的比例
        mask_ratio_class = px / dataset_stats['total_mask_pixels'] * 100
        
        # 计算每张图像中该类别的平均比例
        if img_count > 0:
            avg_ratio_per_img = px / (dataset_stats['total_pixels'] / num_images) / img_count * 100
        else:
            avg_ratio_per_img = 0.0
            
        # 计算包含该类别的图像比例
        pct_images = img_count / num_images * 100
        
        class_stats.append((
            cls, px, mask_ratio_class, avg_ratio_per_img, img_count, pct_images
        ))
    
    # 按像素数排序（从大到小）
    class_stats.sort(key=lambda x: x[1], reverse=True)
    
    # 输出表格
    for stat in class_stats:
        f.write(f"{stat[0]:<15} {stat[1]:<15,} {stat[2]:<12.4f}% {stat[3]:<14.4f}% {stat[4]:<10} {stat[5]:<10.2f}%\n")
    
    # 统计不包含任何对象的图像
    empty_count = sum(1 for stats in dataset_stats['per_class_images'] if stats == 0)
    f.write(f"\nImages without any objects: {empty_count}/{num_images} ({empty_count/num_images*100:.2f}%)\n")

print(f"可视化图像已保存至: {output_dir}")
print(f"单图统计已保存至: {os.path.join(stats_dir, 'per_image_stats.csv')}")
print(f"数据集统计已保存至: {os.path.join(stats_dir, 'dataset_summary.txt')}")