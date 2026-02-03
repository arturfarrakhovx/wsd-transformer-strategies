import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import GroupShuffleSplit
import evaluate

# --- 1. Dataset Class ---
class GlossDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = row['context']
        gloss = row['gloss']
        label = row['label']

        # Prepare input as pairs: [CLS] Context [SEP] Gloss [SEP]
        encoding = self.tokenizer(
            context,
            gloss,
            truncation=True,
            max_length=self.max_len,
            padding=False, # Padding handled by collator for efficiency
            return_tensors=None # Return lists, let collator tensorize
        )

        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': int(label)
        }
        
        # If token_type_ids exist, include them
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids']

        return item

# --- 2. Metrics ---
metric_acc = evaluate.load("accuracy")
metric_p = evaluate.load("precision")
metric_r = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)
    p = metric_p.compute(predictions=predictions, references=labels)
    r = metric_r.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    
    return {
        "accuracy": acc["accuracy"],
        "precision": p["precision"],
        "recall": r["recall"],
        "f1": f1["f1"]
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get labels
        labels = inputs.get("labels")
        
        # Run model
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Zeros are about 8-10 times more than ones.
        # pos_weight=10.0 will make the model "fear" missing a 1.
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(model.device))
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 3. Main Training ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the GLOSS parquet file")
    parser.add_argument("--output_dir", type=str, default="./deberta_wsd_gloss")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    args = parser.parse_args()

    # Load Data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_parquet(args.data_file)
    
    # --- Split Data preserving Groups ---
    # Split by 'instance_group_id' so all candidates for one word stay together
    print("Splitting data by instance groups...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=111)
    train_idx, val_idx = next(gss.split(df, groups=df['instance_group_id']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Train size: {len(train_df)} rows")
    print(f"Val size: {len(val_df)} rows")

    # Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add the special token [TGT] to the tokenizer
    special_tokens_dict = {'additional_special_tokens': ['[TGT]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens: [TGT]")

    train_dataset = GlossDataset(train_df, tokenizer)
    val_dataset = GlossDataset(val_df, tokenizer)
    
    # Data Collator (Dynamic Padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model Initialization
    # Using 2 labels: 0 (False Gloss), 1 (True Gloss)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    # Resize embeddings because [TGT] token added
    model.resize_token_embeddings(len(tokenizer))

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1000,
        disable_tqdm=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=6,
        remove_unused_columns=False
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
