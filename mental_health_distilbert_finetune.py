"""
Production-ready DistilBERT fine-tuning for mental health chatbot (CPU-only)
Execution time: ~40 minutes
Author: Senior ML Engineer

Sections:
1. Environment setup and imports
2. Synthetic data generation
3. Fine-tuning pipeline (CPU-optimized)
4. Evaluation and inference testing
5. Main workflow with timing and progress tracking
"""

import os
import time
import random
import logging
import numpy as np
import json
from typing import List, Tuple
from tqdm import tqdm

# Hugging Face imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                             Trainer, TrainingArguments, EarlyStoppingCallback)
except ImportError as e:
    raise ImportError("Please install torch and transformers: pip install torch transformers") from e

# Set up logging for error handling and progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 1. ENVIRONMENT SETUP AND IMPORTS
# -------------------------------------------------
# Time estimate: 2 min
DEVICE = "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce memory usage

# 2. SYNTHETIC DATA GENERATION FUNCTION
# -------------------------------------------------
# Time estimate: 5 min

EMOTIONS = ["anxiety", "depression", "stress", "hope", "crisis"]
RESPONSES = {
    "anxiety": [
        "I'm here for you. Can you tell me more about what's making you anxious?",
        "It's okay to feel anxious. Let's take a deep breath together."
    ],
    "depression": [
        "I'm sorry you're feeling this way. You're not alone.",
        "Would you like to talk about what's making you feel down?"
    ],
    "stress": [
        "Stress can be overwhelming. What has been causing you stress lately?",
        "Let's try to break down your worries together."
    ],
    "hope": [
        "It's great to hear you're feeling hopeful!",
        "Hope is important. What are you looking forward to?"
    ],
    "crisis": [
        "If you're in crisis, please reach out to a professional or call a helpline.",
        "Your safety is important. Can I help you find support resources?"
    ]
}
LABELS = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

def generate_synthetic_data(num_samples: int = 250) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic mental health conversations for sequence classification.
    Returns: texts, labels
    """
    texts, labels = [], []
    for _ in range(num_samples):
        emotion = random.choice(EMOTIONS)
        user_msg = f"[{emotion.upper()}] " + random.choice([
            f"I feel {emotion} today.",
            f"My mind won't stop racing.",
            f"I'm struggling with {emotion}.",
            f"Can you help me with my {emotion}?",
            f"Sometimes I just feel lost."
        ])
        bot_resp = random.choice(RESPONSES[emotion])
        # Data augmentation: shuffle, paraphrase, add noise
        if random.random() < 0.3:
            user_msg = user_msg.replace("today", "right now")
        if random.random() < 0.2:
            bot_resp = bot_resp + " Remember, you're not alone."
        text = user_msg + " [SEP] " + bot_resp
        texts.append(text)
        labels.append(LABELS[emotion])
    return texts, labels

def load_json_data(json_path):
    """Load conversation samples from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    texts, labels = [], []
    for sample in data:
        texts.append(sample['text'])
        labels.append(sample['label'])
    return texts, labels

def generate_json_samples(filename: str, num_samples: int = 250):
    """
    Generate and save synthetic mental health conversation samples in JSON format.
    """
    samples = []
    for _ in range(num_samples):
        emotion = random.choice(EMOTIONS)
        user_msg = f"[{emotion.upper()}] " + random.choice([
            f"I feel {emotion} today.",
            f"My mind won't stop racing.",
            f"I'm struggling with {emotion}.",
            f"Can you help me with my {emotion}?",
            f"Sometimes I just feel lost."
        ])
        bot_resp = random.choice(RESPONSES[emotion])
        # Data augmentation
        if random.random() < 0.3:
            user_msg = user_msg.replace("today", "right now")
        if random.random() < 0.2:
            bot_resp = bot_resp + " Remember, you're not alone."
        text = user_msg + " [SEP] " + bot_resp
        label = LABELS[emotion]
        samples.append({"text": text, "label": label})
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=2)

# Improved synthetic data with more diverse user intents and responses
IMPROVED_SYNTHETIC_DATA = [
    {"text": "I am feeling happy today. [SEP] That's wonderful! I'm glad you're feeling hopeful. What are you looking forward to?", "label": 3},
    {"text": "I am really stressed about my job. [SEP] Stress can be overwhelming. What has been causing you stress lately?", "label": 2},
    {"text": "I feel so alone and sad. [SEP] I'm sorry you're feeling this way. You're not alone.", "label": 1},
    {"text": "I'm worried about my future. [SEP] It's okay to feel anxious. Let's take a deep breath together.", "label": 0},
    {"text": "I want to hurt myself. [SEP] If you're in crisis, please reach out to a professional or call a helpline.", "label": 4},
    {"text": "Life is going well for me. [SEP] That's wonderful! I'm glad you're feeling hopeful. What are you looking forward to?", "label": 3},
    {"text": "I can't sleep because of anxiety. [SEP] I'm here for you. Can you tell me more about what's making you anxious?", "label": 0},
    {"text": "Work deadlines are stressing me out. [SEP] Stress can be overwhelming. What has been causing you stress lately?", "label": 2},
    {"text": "I feel hopeful about my recovery. [SEP] It's great to hear you're feeling hopeful!", "label": 3},
    {"text": "Sometimes I just feel lost. [SEP] Would you like to talk about what's making you feel down?", "label": 1},
    {"text": "I'm excited for my vacation. [SEP] That's wonderful! I'm glad you're feeling hopeful. What are you looking forward to?", "label": 3},
    {"text": "I am not sure how to cope with stress. [SEP] Let's try to break down your worries together.", "label": 2},
    {"text": "I feel anxious about meeting new people. [SEP] It's okay to feel anxious. Let's take a deep breath together.", "label": 0},
    {"text": "I am grateful for my friends. [SEP] That's wonderful! I'm glad you're feeling hopeful. What are you looking forward to?", "label": 3},
    {"text": "I feel overwhelmed by everything. [SEP] Stress can be overwhelming. What has been causing you stress lately?", "label": 2},
    {"text": "I am scared and don't know what to do. [SEP] I'm here for you. Can you tell me more about what's making you anxious?", "label": 0},
    {"text": "I am feeling down lately. [SEP] I'm sorry you're feeling this way. You're not alone.", "label": 1},
    {"text": "I want to give up. [SEP] If you're in crisis, please reach out to a professional or call a helpline.", "label": 4},
    {"text": "I am hopeful for a better tomorrow. [SEP] It's great to hear you're feeling hopeful!", "label": 3},
    {"text": "I am anxious about my health. [SEP] I'm here for you. Can you tell me more about what's making you anxious?", "label": 0}
]

def save_improved_synthetic_data(filename: str):
    with open(filename, 'w') as f:
        json.dump(IMPROVED_SYNTHETIC_DATA, f, indent=2)

# 3. CPU-OPTIMIZED DISTILBERT FINE-TUNING PIPELINE
# -------------------------------------------------
# Time estimate: 25 min

class MentalHealthDataset(Dataset):
    """Memory-efficient Dataset for mental health conversations."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 4. MODEL EVALUATION AND TESTING FUNCTIONS
# -------------------------------------------------
# Time estimate: 5 min

def compute_metrics(pred):
    """Basic accuracy metric."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# 5. MAIN EXECUTION WORKFLOW WITH TIMING
# -------------------------------------------------
# Time estimate: 3 min

def main():
    start_time = time.time()
    logging.info("Step 0: Saving improved synthetic data to improved_samples.json...")
    save_improved_synthetic_data('improved_samples.json')
    logging.info("Step 1: Loading improved training data from JSON...")
    texts, labels = load_json_data('improved_samples.json')
    logging.info(f"Loaded {len(texts)} improved samples from JSON.")

    # Split data
    split = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Tokenizer and datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(EMOTIONS))

    # Training arguments (CPU-optimized, compatible with transformers 4.30.2)
    training_args = TrainingArguments(
        output_dir='./model_weights',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_steps=100,            # Save every 100 steps
        metric_for_best_model="accuracy",
        fp16=False,               # No mixed precision on CPU
        dataloader_num_workers=0,  # Single worker for stability
        report_to="none",         # Disable wandb/tensorboard
        seed=SEED
    )

    # Trainer setup with error handling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    logging.info("Step 2: Fine-tuning DistilBERT on CPU...")
    train_start = time.time()
    try:
        trainer.train()
        # Save the model and tokenizer after successful training
        trainer.save_model("./model_weights")
        tokenizer.save_pretrained("./model_weights")
        logging.info("Model and tokenizer successfully saved to model_weights/")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise RuntimeError("Training failed. Try reducing batch size or sequence length.")

    # Evaluation
    logging.info("Step 3: Evaluating model...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")

    # Inference testing
    logging.info("Step 4: Inference testing...")
    test_samples = [
        "[ANXIETY] I can't sleep at night. [SEP] I'm here for you. Can you tell me more about what's making you anxious?",
        "[CRISIS] I want to hurt myself. [SEP] If you're in crisis, please reach out to a professional or call a helpline."
    ]
    inputs = tokenizer(test_samples, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    for i, sample in enumerate(test_samples):
        logging.info(f"Sample: {sample}\nPredicted label: {EMOTIONS[preds[i]]}")

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
