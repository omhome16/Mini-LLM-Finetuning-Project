import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import math

from ..models.transformer import CausalLM
from ..utils.tokenizer import load_pythia_tokenizer
from ..data.dataset import InstructionDataset

def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        epochs: int,
        lr: float,
        device: str,
        checkpoint_dir: str = 'checkpoints'
):

    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            _, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"\n--- Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f} ---")

        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")