# Chinese-Name-Classifier
A project to classify whether a given name is a **Chinese name written in Pinyin** or not, using **BERT + LoRA fine-tuning**. Such as "Jun Jiang" is a Chinese Name written in Pinyin, but "Jo Jung" is not.

## Features
- Fine-tuning `bert-base-uncased` with LoRA adapters (PEFT)
- Synthetic dataset generation (Chinese surnames + given names vs non-Chinese names)
- Training pipeline with Hugging Face `Trainer`
- Inference script for quick testing

## Quickstart
```bash
git clone https://github.com/yourname/chinese-name-classifier.git
cd chinese-name-classifier
pip install -r requirements.txt
python train.py
python infer.py
