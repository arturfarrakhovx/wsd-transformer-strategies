import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DebertaV2PreTrainedModel, 
    DebertaV2Config
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
import evaluate

# --- 1. Custom Dataset Class ---
class WSDDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row['sentence']
        c_start = row['char_start']
        # c_end = row['char_end'] # Not strictly needed for logic, start is enough

        # Tokenization with offset mapping
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding['offset_mapping'].squeeze().tolist()
        target_token_idx = 0

        # Find the token corresponding to the word start
        for i, (o_start, o_end) in enumerate(offsets):
            if o_start == 0 and o_end == 0: continue 

            if o_start == c_start:
                target_token_idx = i
                break

            if o_start < c_start and o_end > c_start:
                 target_token_idx = i
                 break

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target_token_idx': torch.tensor(target_token_idx, dtype=torch.long),
            'labels': torch.tensor(row['label_id'], dtype=torch.long)
        }

        return item

# --- 2. Custom Model Class ---
class DebertaV3ForWSD(DebertaV2PreTrainedModel): 
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, target_token_idx=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state

        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        
        # Extract vector for the specific target token
        target_vectors = sequence_output[batch_indices, target_token_idx]

        target_vectors = self.dropout(target_vectors)
        logits = self.classifier(target_vectors)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# --- 3. Main Execution Function ---
def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa V3 for WSD")
    parser.add_argument("--data_file", type=str, required=True, help="Path to parquet data file")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label map json")
    parser.add_argument("--output_dir", type=str, default="./deberta_wsd_custom", help="Directory to save model")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_parquet(args.data_file)
    
    with open(args.label_map, 'r') as f:
        label2id = json.load(f)
    num_labels = len(label2id)
    print(f"Total classes: {num_labels}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=111)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = WSDDataset(train_df, tokenizer)
    val_dataset = WSDDataset(val_df, tokenizer)

    # Model Initialization
    print(f"Initializing model: {args.model_name}")
    model = DebertaV3ForWSD.from_pretrained(args.model_name, num_labels=num_labels)

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        disable_tqdm=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
