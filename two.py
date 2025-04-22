from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from coco_loader import create_coco_loader
import torch
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict
import torchvision

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    Box format: [x1, y1, x2, y2]
    """
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def convert_to_xywh(box):
    """Convert box from [x1, y1, x2, y2] to [x, y, width, height]"""
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

def convert_to_xyxy(box):
    """Convert box from [x, y, width, height] to [x1, y1, x2, y2]"""
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

def compute_ap(precision, recall):
    """Compute Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap

def evaluate_detections(predictions, ground_truth, iou_threshold=0.5, max_dets=100):

    # Group predictions and ground truth by image_id and category_id
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(lambda: defaultdict(list))
    
    # Get all category ids
    category_ids = set()
    
    # Process predictions
    for pred in predictions:
        img_id = pred['image_id']
        cat_id = pred['category_id']
        pred_by_img[img_id].append(pred)
        category_ids.add(cat_id)
    
    # Process ground truth
    for gt in ground_truth:
        img_id = gt['image_id']
        cat_id = gt['category_id']
        gt_by_img[img_id][cat_id].append(gt)
        category_ids.add(cat_id)
    
    # Metrics storage
    metrics = {
        'precision': {},
        'recall': {},
        'ap': {},
        'f1_score': {}
    }
    
    # Calculate metrics for each category
    for cat_id in category_ids:
        true_positives = []
        false_positives = []
        scores = []
        num_gt = 0
        
        # Count total ground truth for this category
        for img_id in gt_by_img:
            num_gt += len(gt_by_img[img_id].get(cat_id, []))
        
        # Process each image
        for img_id in pred_by_img:
            # Get predictions for this image
            img_preds = [p for p in pred_by_img[img_id] if p['category_id'] == cat_id]
            
            # Sort predictions by score in descending order
            img_preds = sorted(img_preds, key=lambda x: x['score'], reverse=True)
            
            # Limit number of detections
            img_preds = img_preds[:max_dets]
            
            # Get ground truth for this image and category
            img_gts = gt_by_img[img_id].get(cat_id, [])
            
            # Mark each ground truth as matched or not
            matched_gt = [False] * len(img_gts)
            
            # Check each prediction
            for pred in img_preds:
                pred_bbox = convert_to_xyxy(pred['bbox'])
                pred_score = pred['score']
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt in enumerate(img_gts):
                    if matched_gt[gt_idx]:
                        continue
                        
                    gt_bbox = convert_to_xyxy(gt['bbox'])
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if we have a match
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    true_positives.append(1)
                    false_positives.append(0)
                    matched_gt[best_gt_idx] = True
                else:
                    true_positives.append(0)
                    false_positives.append(1)
                
                scores.append(pred_score)
        
        # Sort by score
        inds = np.argsort(scores)[::-1]
        true_positives = np.array(true_positives)[inds]
        false_positives = np.array(false_positives)[inds]
        
        # Compute cumulative sum
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        # Compute precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
        recall = tp_cumsum / (num_gt + np.finfo(float).eps)
        
        # Compute AP
        ap = compute_ap(precision, recall)
        
        # Store metrics
        metrics['precision'][cat_id] = precision
        metrics['recall'][cat_id] = recall
        metrics['ap'][cat_id] = ap
        
        # Compute F1 score (optional)
        if len(precision) > 0 and len(recall) > 0:
            f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
            metrics['f1_score'][cat_id] = np.max(f1)
        else:
            metrics['f1_score'][cat_id] = 0.0
    
    # Compute mAP
    metrics['mAP'] = np.mean([metrics['ap'][cat_id] for cat_id in metrics['ap']])
    
    return metrics

def main():
    # Initialize model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    weights= FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    # model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    
    # Initialize preprocessing
    preprocess = weights.transforms()
    
    # Load COCO dataset
    images_dir = "./coco_data/val2017"
    annotations_file = "./coco_data/annotations/instances_val2017.json"
    loader, dataset = create_coco_loader(images_dir, annotations_file)
    
    # Run inference
    results = []
    ground_truth = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm(dataset)):
            if i >= 500:  # Limit to first 100 images for testing
                break
                
            image_id = target['image_id'].item()
            
            # Add ground truth for this image
            for obj_idx in range(len(target['boxes'])):
                gt = {
                    'image_id': image_id,
                    'category_id': target['labels'][obj_idx].item(),
                    'bbox': convert_to_xywh(target['boxes'][obj_idx].tolist()),
                    'iscrowd': 0
                }
                ground_truth.append(gt)
            
            # Run model
            batch = [preprocess(img)]
            prediction = model(batch)[0]
            
            # Extract predictions
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Convert predictions to COCO format
            for box, score, label in zip(boxes, scores, labels):
                if score >= 0.05:  # Minimum score threshold
                    # print(score)
                    x1, y1, x2, y2 = box.tolist()
                    coco_box = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                    
                    # Map label to category id
                    category_id = label.item()
                    
                    results.append({
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': coco_box,
                        'score': float(score)
                    })
    
    # Save results to file
    with open('custom_results.json', 'w') as f:
        json.dump(results, f)
    
    # Run custom evaluation
    print("Running custom evaluation...")
    metrics = evaluate_detections(results, ground_truth)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"mAP @ IoU={0.5}: {metrics['mAP']:.4f}")
    
    # Print AP for each category
    print("\nAP by category:")
    for cat_id in sorted(metrics['ap'].keys()):
        print(f"Category {cat_id}: {metrics['ap'][cat_id]:.4f}")
    
    # Optional: Plot precision-recall curves
    try:
        import matplotlib.pyplot as plt
        
        # Plot precision-recall curve for first 5 categories
        plt.figure(figsize=(10, 8))
        
        for i, cat_id in enumerate(sorted(list(metrics['precision'].keys()))[:5]):
            if len(metrics['precision'][cat_id]) > 0 and len(metrics['recall'][cat_id]) > 0:
                plt.plot(
                    metrics['recall'][cat_id], 
                    metrics['precision'][cat_id],
                    label=f'Category {cat_id} (AP: {metrics["ap"][cat_id]:.4f})'
                )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid()
        plt.savefig('precision_recall_curves.png')
        plt.close()
        print("Saved precision-recall curves to 'precision_recall_curves.png'")
    except ImportError:
        print("Matplotlib not installed. Skipping precision-recall curve plotting.")

if __name__ == "__main__":
    main()