# --- Fine-Tuning GPT-Neo 1.3B with LoRA, BitFit, and Adapters, then Comparing Attention ---

import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AdamW
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Set up accelerator for MPS
accelerator = Accelerator()
device = accelerator.device

# 1) Model info
model_name = "EleutherAI/gpt-neo-1.3B"#"microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.save_pretrained("./patched-tokenizer")

# Reload patched tokenizer
tokenizer = AutoTokenizer.from_pretrained("./patched-tokenizer")

# 2) Load dataset & shuffle
dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
dataset = dataset.shuffle(seed=42)

def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Fine-Tuning Strategies ===

def run_lora():
    """Sets up a LoRA-wrapped model using PEFT."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    base_model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(base_model, peft_config)
    return model, "phi1.5-lora"

def run_bitfit():
    """Freezes all but bias terms in GPT-Neo."""
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model.resize_token_embeddings(len(tokenizer))
    for name, param in model.named_parameters():
        param.requires_grad = ("bias" in name)
    return model, "phi1.5-bitfit"

def freeze_all_but_last(model, last_layers_to_unfreeze=1):
    """
    Finds the max transformer.h.X index in phi1.5 style,
    and only unfreezes that block + lm_head for the last <last_layers_to_unfreeze> blocks.
    """
    max_block_idx = -1
    for name, _ in model.named_parameters():
        if "transformer.h." in name:
            # parse the number after 'transformer.h.'
            layer_str = name.split("transformer.h.")[1].split(".")[0]
            try:
                idx = int(layer_str)
                if idx > max_block_idx:
                    max_block_idx = idx
            except ValueError:
                pass

    min_block_to_unfreeze = max_block_idx - (last_layers_to_unfreeze - 1)
    if min_block_to_unfreeze < 0:
        min_block_to_unfreeze = 0

    print(f"[INFO] Freezing all blocks except h.{min_block_to_unfreeze} to h.{max_block_idx} plus lm_head.")
    for name, param in model.named_parameters():
        if "transformer.h." in name:
            layer_str = name.split("transformer.h.")[1].split(".")[0]
            try:
                idx = int(layer_str)
                if min_block_to_unfreeze <= idx <= max_block_idx:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            except:
                param.requires_grad = False
        elif "lm_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def run_adapter():
    """Freeze all layers but last few blocks + lm_head."""
    adapter_model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    adapter_model.resize_token_embeddings(len(tokenizer))
    # Freeze everything except last 3 blocks + lm_head
    adapter_model = freeze_all_but_last(adapter_model, last_layers_to_unfreeze=3)
    return adapter_model, "phi1.5-adapter"

def fine_tune_and_save(model, label):
    """
    If a fine-tuned model folder already exists, skip training. Otherwise, train and save.
    """
    # If folder is present, skip training
    if os.path.isdir(f"./{label}"):
        print(f"[INFO] Using existing checkpoint for {label}, skipping training...")
        return

    gc.collect()
    torch.mps.empty_cache()

    model = accelerator.prepare(model)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
    dataloader = accelerator.prepare(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Fine-tuning ({label})", leave=True)
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"]

        outputs = model(**batch)
        loss = outputs.loss
        progress_bar.set_postfix({"loss": loss.item()})

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        torch.mps.empty_cache()
        gc.collect()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"./{label}")
    tokenizer.save_pretrained(f"./{label}")

# 1) LoRA
lora_model, lora_label = run_lora()
fine_tune_and_save(lora_model, lora_label)

# 2) BitFit
bitfit_model, bitfit_label = run_bitfit()
fine_tune_and_save(bitfit_model, bitfit_label)

# 3) Adapter
adapter_model, adapter_label = run_adapter()
fine_tune_and_save(adapter_model, adapter_label)


# === Attention Comparison (Original vs LoRA vs BitFit vs Adapter) ===
text = "The company's stock surged after the earnings report."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}

attn_matrices = {}

# Original
device = "cpu"
gc.collect()
torch.mps.empty_cache()
orig_model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
orig_model.resize_token_embeddings(len(tokenizer))
orig_model = orig_model.to(device)

with torch.no_grad():
    outputs_orig = orig_model(**inputs)
attn_matrices["original"] = outputs_orig.attentions[-1][0, 0].cpu().numpy()

del orig_model, outputs_orig
gc.collect()
torch.mps.empty_cache()

# LoRA
lora_base_model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
lora_base_model.resize_token_embeddings(len(tokenizer))
lora_model_loaded = PeftModel.from_pretrained(lora_base_model, f"./{lora_label}").to(device)

with torch.no_grad():
    outputs_lora = lora_model_loaded(**inputs)
attn_matrices["lora"] = outputs_lora.attentions[-1][0, 0].cpu().numpy()

del lora_base_model, lora_model_loaded, outputs_lora
gc.collect()
torch.mps.empty_cache()

# BitFit
bitfit_model_loaded = AutoModelForCausalLM.from_pretrained(f"./{bitfit_label}", output_attentions=True).to(device)

with torch.no_grad():
    outputs_bitfit = bitfit_model_loaded(**inputs)
attn_matrices["bitfit"] = outputs_bitfit.attentions[-1][0, 0].cpu().numpy()

del bitfit_model_loaded, outputs_bitfit
gc.collect()
torch.mps.empty_cache()

# Adapter
adapter_model_loaded = AutoModelForCausalLM.from_pretrained(f"./{adapter_label}", output_attentions=True)
adapter_model_loaded.resize_token_embeddings(len(tokenizer))
adapter_model_loaded = adapter_model_loaded.to(device)

with torch.no_grad():
    outputs_adapter = adapter_model_loaded(**inputs)
attn_matrices["adapter"] = outputs_adapter.attentions[-1][0, 0].cpu().numpy()

del adapter_model_loaded, outputs_adapter
gc.collect()
torch.mps.empty_cache()

# Plot side by side
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
plot_tokens = [t.replace("Ġ", "▁") for t in tokens]  # or however you want to handle 'Ġ'

# Create a 2x2 grid
fig, axes = plt.subplots(1, 4, figsize=(16, 6))  # Wider figure, shorter height

for i, (label, attn) in enumerate(attn_matrices.items()):
    ax = axes[i]
    sns.heatmap(
        attn,
        xticklabels=plot_tokens,
        yticklabels=plot_tokens,
        cmap="viridis",
        ax=ax,
        cbar=False  # Optional: remove color bar to save space
    )
    ax.set_title(label.capitalize(), fontsize=14, pad=10)
    ax.set_xticklabels(plot_tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(plot_tokens, rotation=0, fontsize=10)

plt.tight_layout()
plt.show()