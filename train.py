# train.py  —— 纯拼音中文名字二分类（LoRA + BERT），无 evaluate 依赖
import os, json, random, torch, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score  # ✅ 用 sklearn 做指标

BASE_MODEL = "bert-base-uncased"
SAVE_DIR = "save/lora_adapter"
TOK_DIR  = "save/tokenizer"
NUM_LABELS = 2
MAX_LEN = 32

id2label = {0:"non_zh", 1:"zh_pinyin"}
label2id = {v:k for k,v in id2label.items()}

# ------------------- 1) 合成数据 -------------------
os.makedirs("data", exist_ok=True)

cn_surnames = ["Wang","Li","Zhang","Liu","Chen","Yang","Zhao","Huang","Zhou","Wu"]
cn_given = ["Wei","Na","Lei","Jun","Hao","Xin","Jing","Yan","Yue","Xuan"]

nonzh_first = ["Anna","Michael","Peter","Maria","Sophia","Thomas","Paul","Lukas"]
nonzh_last  = ["Miller","Smith","Johnson","Brown","Schmidt","Fischer"]

def synth_cn(n):
    out=[]
    for _ in range(n):
        s, g = random.choice(cn_surnames), random.choice(cn_given)
        form = random.choice([f"{s} {g}", f"{g} {s}", f"{s}{g}", f"{s}-{g}"])
        out.append({"text":form, "label":1})
    return out

def synth_non(n):
    out=[]
    for _ in range(n):
        f, l = random.choice(nonzh_first), random.choice(nonzh_last)
        form = random.choice([f"{f} {l}", f"{l} {f}", f"{f}-{l}"])
        out.append({"text":form, "label":0})
    return out

train = synth_cn(800) + synth_non(800)
valid = synth_cn(200) + synth_non(200)
random.shuffle(train); random.shuffle(valid)

with open("data/train.jsonl","w") as f:
    for r in train: f.write(json.dumps(r)+"\n")
with open("data/valid.jsonl","w") as f:
    for r in valid: f.write(json.dumps(r)+"\n")

# ------------------- 2) Tokenizer & 数据 -------------------
ds = load_dataset("json", data_files={"train":"data/train.jsonl","validation":"data/valid.jsonl"})
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

def preprocess(ex):
    out = tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = ex["label"]
    return out

ds_tok = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

# ------------------- 3) 模型 + LoRA -------------------
gc.collect(); torch.cuda.empty_cache()

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
).to(dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

peft_cfg = LoraConfig(
    task_type="SEQ_CLS",
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query","key","value","dense"]  # 适配BERT族
)
model = get_peft_model(base_model, peft_cfg)

# ------------------- 4) 指标（sklearn） -------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }

# ------------------- 5) 训练 -------------------
args = TrainingArguments(
    output_dir="save",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# ------------------- 6) 保存 -------------------
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TOK_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tok.save_pretrained(TOK_DIR)
print("✅ 模型保存到:", SAVE_DIR)
