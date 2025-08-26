# --- I AM COMPARING THE BERT WITH THE FINE_TUNED OPTION


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
SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"


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

        # Update the KB JSON file with new entries
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

# --- Step 1: SBERT decides first ---
print("Finding best match in KB using SBERT...")
query_embedding = sbert_model.encode(user_input, convert_to_tensor=True)

scores = []
for item, details in kb.items():
    full_text = item + " " + details.get("visual_description", "") + " " + details.get("usage", "")
    item_embedding = sbert_model.encode(full_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_embedding, item_embedding).item()
    scores.append((item, score))

scores.sort(key=lambda x: x[1], reverse=True)
top_item, top_score = scores[0]

print("\nSBERT similarity scores:")
for obj, sc in scores[:5]:
    print(f"  {obj}: {sc:.3f}")
print(f"\nTop SBERT match: '{top_item}' with score {top_score:.3f}")

# --- Step 2: Decide action ---
if top_score >= 0.6:
    # Confident → use SBERT match directly
    final_classes = [top_item]
    print(f"SBERT confident — using '{top_item}' without YOLO check.")
else:
    # Not confident → YOLO verifies
    print("SBERT not confident — checking with YOLO...")
    all_results = model.predict(source=IMAGE_PATH, save=False, device=device, conf=0.05)
    visible_objects = {all_results[0].names[int(box.cls)] for box in all_results[0].boxes}

    if top_item in visible_objects:
        final_classes = [top_item]
        print(f"YOLO verified '{top_item}' is visible.")
    else:
        print("YOLO could not verify object — using GPT fallback.")
        final_classes = gpt_fallback(user_input)

# --- Validate final classes ---
final_classes = [cls.strip() for cls in final_classes if isinstance(cls, str) and len(cls.strip()) > 0]
if not final_classes:
    print("No valid classes to detect. Exiting.")
    exit()

print(f"Final detection class: {final_classes}")

# --- Step 3: YOLO detection (final classes only) ---
try:
    model.set_classes(final_classes)
    results = model.predict(source=IMAGE_PATH, save=False, device=device, conf=0.25)
    annotated_image = results[0].plot()

    cv2.imwrite(OUTPUT_PATH, annotated_image)
    print(f"\nDetection complete! Saved to: {OUTPUT_PATH}")

except Exception as e:
    print(f"YOLO detection failed: {e}")
    exit()

# --- Step 4: Print descriptions ---
detected_objects = {results[0].names[int(box.cls)] for box in results[0].boxes}

print("\nDetected object descriptions:")
for obj in detected_objects:
    if obj in kb:
        desc = kb[obj].get("visual_description", "No description")
        print(f"• {obj}: {desc}")
    else:
        print(f"• {obj}: (no KB entry)")
