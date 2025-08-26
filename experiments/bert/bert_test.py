from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
sbert_model = SentenceTransformer("/home/iibrohimm/project/try/yoloWorld/bert/new_bert_st")

# Encode a sample prompt
sentence = "I want to dry my hands"
embedding = sbert_model.encode(sentence)

print("Sentence:", sentence)
print("Embedding length:", len(embedding))
print("Embedding vector:", embedding)
