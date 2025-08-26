import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# --- Select GPU (force GPU 2 here) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# --- CONFIG ---
KB_PATH = "/home/iibrohimm/project/try/yoloWorld/assets/new.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"   # Sentence-BERT style pretrained model
OUTPUT_DIR = "/home/iibrohimm/project/try/yoloWorld/bert/new_bert"

# --- Load KB ---
with open(KB_PATH, "r") as f:
    kb = json.load(f)

# --- Build dataset (vague prompt → label) ---
texts, labels = [], []
label2id, id2label = {}, {}
label_counter = 0

for obj, details in kb.items():
    # Collect all text samples for this class
    vp = details.get("ambiguous_prompts", [])
    vd = [details["visual_description"]] if "visual_description" in details else []
    us = [details["usage"]] if "usage" in details else []
    samples = vp + vd + us

    if not samples:
        continue

    if obj not in label2id:
        label2id[obj] = label_counter
        id2label[label_counter] = obj
        label_counter += 1

    for text in samples:
        texts.append(text)
        labels.append(label2id[obj])


print(f"✅ Dataset built: {len(texts)} samples, {len(label2id)} classes")

# --- Convert to HuggingFace Dataset ---
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- Load tokenizer & model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id
)

# --- Tokenize ---
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --- Training arguments ---
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_eval=True,
    eval_strategy="epoch",   # 👈 evaluate every epoch
    save_strategy="epoch",   # 👈 save every epoch
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=2
)

# --- Define compute metrics ---
import evaluate
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- Train ---
trainer.train()

# --- Final Evaluation ---
metrics = trainer.evaluate()
print("📊 Final Evaluation:", metrics)

# --- Save model ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 Fine-tuned BGE saved at: {OUTPUT_DIR}")
