import json

# Load JSON
with open("/home/iibrohimm/project/try/yoloWorld/assets/KB.json", "r") as f:
    data = json.load(f)

# Total number of object names
total_items = len(data)

# Print each object name
for item_name in data.keys():
    print(item_name)

# Print total count

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
import openai
import cv2
import os
import json
import torch
import warnings

# --- CONFIG ---
warnings.filterwarnings("ignore", category=FutureWarning)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in environment variables.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

YOLO_MODEL_PATH = "/home/iibrohimm/project/try/yoloWorld/PTs/yolov8x-worldv2.pt"
KB_PATH = "/home/iibrohimm/project/try/yoloWorld/assets/KB.json"
IMAGE_PATH = "/home/iibrohimm/project/try/yoloWorld/data/kitchen.png"
OUTPUT_PATH = "/home/iibrohimm/project/try/yoloWorld/results/yolo_result.jpg"
SBERT_MODEL_NAME = "/home/iibrohimm/project/try/yoloWorld/bert/bert_finetuned"

# --- GPT Fallback Function ---
def gpt_fallback(user_input):
    print("\nUsing GPT fallback for object detection...")

    prompt = f"""
You are a smart assistant that helps detect objects in images based on vague user prompts.

The user said: "{user_input}"

1. First, identify 1–3 specific objects the user likely meant.
2. Then, for each object, create a JSON entry with:
    - "visual_description": What it looks like
    - "usage": What it’s used for

Return ONLY a single valid JSON dictionary where:
- Each key is the object name 
- Each value is a dictionary with "visual_description" and "usage"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = response.choices[0].message.content.strip()

        if not content.startswith('{'):
            print(f"GPT did not return a JSON object: {content[:50]}...")
            return []

        new_entries = json.loads(content)

        # Update KB with new entries
        with open(KB_PATH, "r+") as f:
            kb = json.load(f)
            kb.update(new_entries)
            f.seek(0)
            json.dump(kb, f, indent=2)
            f.truncate()

        print(f"GPT added {list(new_entries.keys())} to KB.")
        return list(new_entries.keys())

    except json.JSONDecodeError as e:
        print(f"Failed to parse GPT response as JSON: {e}")
        return []
    except Exception as e:
        print(f"GPT fallback failed: {e}")
        return []


# --- Load Models and KB ---
print("Loading models and knowledge base...")
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
model = YOLO(YOLO_MODEL_PATH)

with open(KB_PATH, "r") as f:
    kb = json.load(f)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Get User Input ---
user_input = input("What do you want to detect? (vague prompt allowed): ").strip()
if not user_input:
    print("Empty input. Exiting.")
    exit()

# --- Step 1: SBERT matches against ALL KB entries ---
print("Finding relevant objects using semantic search (full KB)...")
query_embedding = sbert_model.encode(user_input, convert_to_tensor=True)
scores = []

for item, details in kb.items():
    full_text = (
        item + " " +
        details.get("visual_description", "") + " " +
        details.get("usage", "")
    )
    item_embedding = sbert_model.encode(full_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_embedding, item_embedding).item()
    scores.append((item, score))

# Sort and show top results
scores.sort(key=lambda x: x[1], reverse=True)
print("\nSBERT similarity scores (full KB):")
for obj, sc in scores[:5]:  # top 5 for inspection
    print(f"  {obj}: {sc:.3f}")

top_items = [item for item, _ in scores]
top_score = scores[0][1]
print(f"\nTop SBERT match: '{top_items[0]}' with score {top_score:.3f}")

# --- Step 2: Smart fallback ---
if top_score < 0.6:
    print("Confidence too low — using GPT fallback.")
    final_classes = gpt_fallback(user_input)
else:
    final_classes = [top_items[0]]
    print(f"Using KB match: {final_classes}")

# --- Validate & Filter ---
final_classes = [cls.strip() for cls in final_classes if isinstance(cls, str) and len(cls.strip()) > 0]
if not final_classes:
    print("No valid classes to detect. Exiting.")
    exit()

print(f"Final detection class: {final_classes}")

# --- Step 3: YOLO detection ---
try:
    model.set_classes(final_classes)
    results = model.predict(source=IMAGE_PATH, save=False, device=device, conf=0.25)
    annotated_image = results[0].plot()

    # Save annotated image
    cv2.imwrite(OUTPUT_PATH, annotated_image)
    print(f"\nDetection complete! Saved to: {OUTPUT_PATH}")

except Exception as e:
    print(f"YOLO detection failed: {e}")
    exit()

# --- Step 4: Print Detected Object Descriptions ---
detected_objects = set()
for box in results[0].boxes:
    label_idx = int(box.cls)
    label_name = results[0].names[label_idx]
    detected_objects.add(label_name)

print("\nDetected object descriptions:")
for obj in detected_objects:
    if obj in kb:
        desc = kb[obj]["visual_description"]
        print(f"• {obj}: {desc}")
    else:
        print(f"• {obj}:")
