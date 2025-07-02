import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import os
import json
from typing import Dict

from models.transformer import CausalLM, ModelArgs
from utils.tokenizer import load_pythia_tokenizer
from data.dataset import InstructionDataset
from training.trainer import train_model


def load_and_map_weights(custom_model: CausalLM, hf_model_name: str) -> CausalLM:
    """
    Loads pre-trained weights from a Hugging Face model and maps them
    to the parameters of the custom model.
    """
    print(f"Loading pre-trained weights from {hf_model_name}...")

    # 1. Load the pre-trained model from Hugging Face
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_state_dict = hf_model.state_dict()

    # 2. Get your custom model's state dictionary
    custom_state_dict = custom_model.state_dict()

    # 3. Create the key mapping logic
    print("Mapping state dictionary keys...")

    # Pythia has a combined QKV matrix, we need to split it for our model.
    for i in range(custom_model.config.n_layers):
        hf_qkv_key = f'gpt_neox.layers.{i}.attention.query_key_value.weight'

        if hf_qkv_key in hf_state_dict:
            qkv_weight = hf_state_dict[hf_qkv_key]
            # Pythia's QKV is concatenated. Split it based on head_dim * num_heads.
            q_weight, k_weight, v_weight = torch.split(qkv_weight, custom_model.config.dim, dim=0)

            # Map the split tensors to your custom model's layers
            custom_state_dict[f'decoder_blocks.{i}.attention.q_proj.weight'] = q_weight
            custom_state_dict[f'decoder_blocks.{i}.attention.k_proj.weight'] = k_weight
            custom_state_dict[f'decoder_blocks.{i}.attention.v_proj.weight'] = v_weight

            # --- Map other layers ---
            # Attention Norm
            custom_state_dict[f'decoder_blocks.{i}.attention_norm.weight'] = hf_state_dict[
                f'gpt_neox.layers.{i}.input_layernorm.weight']
            # Output projection
            custom_state_dict[f'decoder_blocks.{i}.attention.o_proj.weight'] = hf_state_dict[
                f'gpt_neox.layers.{i}.attention.dense.weight']
            # FFN Norm
            custom_state_dict[f'decoder_blocks.{i}.ffn_norm.weight'] = hf_state_dict[
                f'gpt_neox.layers.{i}.post_attention_layernorm.weight']
            # FFN layers (dense_h_to_4h is fc1, dense_4h_to_h is fc2)
            custom_state_dict[f'decoder_blocks.{i}.feed_forward.fc1.weight'] = hf_state_dict[
                f'gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight']
            custom_state_dict[f'decoder_blocks.{i}.feed_forward.fc1.bias'] = hf_state_dict[
                f'gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias']
            custom_state_dict[f'decoder_blocks.{i}.feed_forward.fc2.weight'] = hf_state_dict[
                f'gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight']
            custom_state_dict[f'decoder_blocks.{i}.feed_forward.fc2.bias'] = hf_state_dict[
                f'gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias']

    # Map final layers
    custom_state_dict['final_norm.weight'] = hf_state_dict['gpt_neox.final_layer_norm.weight']
    custom_state_dict['lm_head.weight'] = hf_state_dict['embed_out.weight']

    # 4. Load the mapped state dictionary with strict=False to ignore mismatches
    print("Loading mapped state dictionary into custom model...")
    custom_model.load_state_dict(custom_state_dict, strict=False)

    print("Weights loaded successfully!")
    return custom_model

def main():
    # 1. Define model parameters
    config = ModelArgs()

    # 2. Prepare directories and dummy data
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'instruction_data.jsonl')

    # 3. Load tokenizer
    tokenizer = load_pythia_tokenizer()

    # 4. Instantiate your custom model and move to device
    model = CausalLM(config).to(config.device)

    # 5. Load pre-trained weights from Hugging Face
    model = load_and_map_weights(model, "EleutherAI/pythia-70m")

    # 6. Load your instruction fine-tuning dataset
    train_dataset = InstructionDataset(data_path, tokenizer, max_seq_len=config.max_seq_len)

    # 7. Create PyTorch DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.max_batch_size,
        shuffle=True,
        drop_last=True
    )

    # 8. Instantiate and run the training loop
    train_model(model, train_dataloader, epochs=config.epochs, lr=1e-4, device=config.device)


if __name__ == "__main__":
    main()