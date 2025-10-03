import torch, os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "bert-base-uncased"
SAVE_DIR = "save/lora_adapter"
TOK_DIR  = "save/tokenizer"

id2label = {0:"non_zh", 1:"zh_pinyin"}
label2id = {v:k for k,v in id2label.items()}

base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, id2label=id2label, label2id=label2id
).to("cuda" if torch.cuda.is_available() else "cpu")

model = PeftModel.from_pretrained(base, SAVE_DIR)
tok   = AutoTokenizer.from_pretrained(TOK_DIR if os.path.exists(TOK_DIR) else BASE_MODEL)

def predict_is_zh_pinyin(name, threshold=0.5):
    batch = tok(name, return_tensors="pt", truncation=True, padding=True, max_length=32).to(model.device)
    with torch.no_grad():
        prob = torch.softmax(model(**batch).logits, dim=-1)[0,1].item()
    return (prob>=threshold), prob

tests = ["Wang Wei","Li-Na","Zhang San","Wei Zhang","Anna Miller","Johannes Mueller","Paul Schmidt","WangWei","Wei-Wang"]
for t in tests:
    print(f"{t} -> {predict_is_zh_pinyin(t)}")
