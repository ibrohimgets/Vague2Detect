import os, json
import torch
from collections import defaultdict
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util

# --- Paths ---
DATASET_PATH = "/home/iibrohimm/project/try/yoloWorld/bench/openImages/mixed_dataset.json"
MODEL_PATH = "/home/iibrohimm/project/try/yoloWorld/PTs/yolov8x-worldv2.pt"
SBERT_PATH = "/home/iibrohimm/project/try/yoloWorld/bert/new_bert_st"  # fine-tuned SBERT

# --- Load models ---
yolo = YOLO(MODEL_PATH)
sbert = SentenceTransformer(SBERT_PATH)

# --- Load dataset ---
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# --- User input ---
user_prompt = input("Enter a vague prompt: ")

# --- Encode prompt once ---
query_emb = sbert.encode(user_prompt, convert_to_tensor=True)

# --- Stats ---
correct_cls, correct_det, total = 0, 0, 0
class_stats = defaultdict(lambda: {"correct_cls": 0, "correct_det": 0, "total": 0})

# --- Evaluation loop ---
for entry in dataset:
    img_path = entry["image"]
    gt = entry["gt"][0]
    gt_cls, gt_bbox = gt["cls"], gt["bbox"]

    # ---- Step 1: SBERT choose best class from dataset classes ----
    # (Compare against all GT classes in dataset, not KB)
    scores = []
    for candidate in {e["gt"][0]["cls"] for e in dataset}:  # unique classes
        cand_emb = sbert.encode(candidate, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_emb, cand_emb).item()
        scores.append((candidate, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    pred_cls = scores[0][0]

    # Count class accuracy (VPSR)
    class_stats[gt_cls]["total"] += 1
    if pred_cls == gt_cls:
        correct_cls += 1
        class_stats[gt_cls]["correct_cls"] += 1

    # ---- Step 2: YOLO detect objects ----
    results = yolo.predict(source=img_path, conf=0.25, device=0, save=False, verbose=False, imgsz=416)

    # Collect YOLO boxes for predicted class
    pred_boxes = [
        box.xyxy[0].cpu().numpy().tolist()
        for box in results[0].boxes
        if results[0].names[int(box.cls)] == pred_cls
    ]

    if pred_boxes:
        correct_det += 1
        class_stats[gt_cls]["correct_det"] += 1

    total += 1

# --- Results ---
print("="*50)
print(f"Prompt: {user_prompt}")
print(f"Total Images: {total}")
print(f"VPSR (class match only): {correct_cls/total:.2f} ({correct_cls}/{total})")
print(f"Detection Success (YOLO saw predicted class): {correct_det/total:.2f} ({correct_det}/{total})")

print("\nPer-class stats:")
for cls, stats in class_stats.items():
    cls_acc = stats["correct_cls"]/stats["total"] if stats["total"] else 0
    det_acc = stats["correct_det"]/stats["total"] if stats["total"] else 0
    print(f"{cls:10s} → VPSR: {cls_acc:.2f}, DetAcc: {det_acc:.2f} ({stats['total']} samples)")
