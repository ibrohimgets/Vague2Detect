import os
import json
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util

# --- Paths ---
DATASET_PATH = "/home/iibrohimm/project/try/yoloWorld/bench/openImages/mixed_dataset.json"
MODEL_PATH = "/home/iibrohimm/project/try/yoloWorld/PTs/yolov8x-worldv2.pt"
SBERT_PATH = SentenceTransformer("/home/iibrohimm/project/try/yoloWorld/bert/new_bert_st")
KB_PATH = "/home/iibrohimm/project/try/yoloWorld/assets/KB.json"

# --- Load models ---
sbert = SBERT_PATH
yolo = YOLO(MODEL_PATH)

# --- Load KB ---
with open(KB_PATH) as f:
    kb = json.load(f)

# --- IoU helper ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# --- Load dataset ---
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# --- Ask for ambiguous prompt ---
user_prompt = input("Enter an ambiguous prompt (e.g., 'something to slice vegetables'): ")

correct_cls, correct_det, total = 0, 0, 0

# --- Stats trackers ---
class_stats = defaultdict(lambda: {"correct_cls": 0, "correct_det": 0, "total": 0})
confusion_counts = defaultdict(int)  # how many times SBERT picked each class

# --- Evaluation loop ---
for entry in dataset:
    img_path = entry["image"]  # already full path from merged JSON
    gt = entry["gt"][0]
    gt_cls, gt_bbox = gt["cls"], gt["bbox"]

    # ---- SBERT predict class ----
    query_emb = sbert.encode(user_prompt, convert_to_tensor=True)
    scores = []
    for item, details in kb.items():
        text = item + " " + details.get("visual_description", "") + " " + details.get("usage", "")
        item_emb = sbert.encode(text, convert_to_tensor=True)
        scores.append((item, util.pytorch_cos_sim(query_emb, item_emb).item()))
    scores.sort(key=lambda x: x[1], reverse=True)
    pred_cls = scores[0][0]

    # 🔎 Show top-2 matches only once
    if total == 0:
        print(f"\n🔎 Top-2 SBERT matches for prompt '{user_prompt}':")
        for label, score in scores[:2]:
            print(f"  {label:<10} → {score:.4f}")

    # Track confusion
    confusion_counts[pred_cls] += 1

    # Count class accuracy
    class_stats[gt_cls]["total"] += 1
    if pred_cls == gt_cls:
        correct_cls += 1
        class_stats[gt_cls]["correct_cls"] += 1

    # ---- YOLO detect (filter by predicted class) ----
    results = yolo.predict(source=img_path, conf=0.25, device=0, save=False)

    pred_boxes = [
        box.xyxy[0].cpu().numpy().tolist()
        for box in results[0].boxes
        if results[0].names[int(box.cls)] == pred_cls
    ]

    if pred_boxes:
        pred_bbox = pred_boxes[0]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_scaled = [gt_bbox[0] * w, gt_bbox[1] * h, gt_bbox[2] * w, gt_bbox[3] * h]

        iou = compute_iou(pred_bbox, gt_scaled)
        if pred_cls == gt_cls and iou >= 0.5:
            correct_det += 1
            class_stats[gt_cls]["correct_det"] += 1

    total += 1

# --- Overall results ---
print(f"\nPrompt used: '{user_prompt}'")
print(f"Class Accuracy: {correct_cls/total:.2f}")
print(f"Detection Accuracy (IoU>=0.5): {correct_det/total:.2f}")

# --- Per-class results ---
print("\nPer-class results:")
for cls, stats in class_stats.items():
    cls_acc = stats["correct_cls"] / stats["total"] if stats["total"] > 0 else 0
    det_acc = stats["correct_det"] / stats["total"] if stats["total"] > 0 else 0
    print(f"{cls:10s} → Class Acc: {cls_acc:.2f}, Detection Acc: {det_acc:.2f} ({stats['total']} samples)")

# --- Confusion summary ---
print("\nSBERT predicted classes distribution:")
for cls, count in confusion_counts.items():
    print(f"{cls:10s} → {count} times")
