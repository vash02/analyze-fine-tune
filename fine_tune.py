import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import tensorboardX

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, TaskType, get_peft_model
from copy import deepcopy
from dataset_loader import DatasetLoader


class MlmFineTuner:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # 1) Load MLM model instead of sequence classification
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2) Save initial attention weights for comparison
        self.initial_weights = self.get_attention_weights()

        # 3) TensorBoard writer
        self.writer = tensorboardX.SummaryWriter(f"runs/MLM-FineTuning-{model_name}")
        self.step = 0

    def get_attention_weights(self):
        """Get only attention weights before fine-tuning."""
        return {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
            if "attention" in name
        }

    def extract_attention_scores(self, model, text):
        """Extract attention scores for a given text input."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        # `outputs.attentions` is a tuple of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
        # We'll just return the last layer for demonstration
        return outputs.attentions[-1], self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())

    def apply_finetuning(self, method="mlm_full"):
        """
        Fine-tune the model on MLM using different parameter-efficient methods (LoRA, BitFit, etc.).
        We'll store the model under ./saved_models/<method>
        """
        model_path = f"./saved_models/{method}"

        # If model already exists, load from disk
        if os.path.exists(model_path):
            print(f"Loading existing model for {method}")
            return AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)

        # Make a fresh copy of the base model
        model = deepcopy(self.model)

        # Parameter-efficient approach
        if method == "mlm_full":
            for param in model.parameters():
                param.requires_grad = True
        elif method == "mlm_freeze_base":
            for name, param in model.named_parameters():
                # Freeze everything except the 'cls' prediction head in MLM
                if "bert" in name or "roberta" in name or "distilbert" in name:
                    param.requires_grad = False
        elif method == "mlm_bitfit":
            for name, param in model.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False
        elif method == "mlm_lora":
            peft_config = LoraConfig(
                task_type=TaskType.MASKED_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
            model = get_peft_model(model, peft_config).to(self.device)

        # Trainer setup
        training_args = TrainingArguments(
            output_dir=model_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=10,
            report_to="tensorboard"
        )

        # Use a DataCollator for Masked Language Modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,  # We'll set self.dataset in evaluate_finetuning
            data_collator=data_collator
        )

        # Fine-tune with MLM
        trainer.train()

        # Save the fine-tuned model
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        print(f"Fine-tuned MLM model saved at: {model_path}")
        return model

    def compare_attention_matrices(self, text, methods):
        """Compare attention weight differences vs. the original (pretrained) model."""
        # 1) Load the original, pretrained model as baseline
        original_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        original_attentions, tokens = self.extract_attention_scores(original_model, text)

        # We'll store attention differences for each method
        attention_matrices = {}
        for method in methods:
            model_path = f"./saved_models/{method}"
            if os.path.exists(model_path):
                # 2) Load fine-tuned MLM model
                ft_model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
                ft_attentions, _ = self.extract_attention_scores(ft_model, text)

                # 3) Compute differences from original
                # We'll compare last-layer attention for demonstration
                diff = torch.abs(ft_attentions - original_attentions).mean(dim=1).squeeze(0).cpu().numpy()
                attention_matrices[method] = diff
            else:
                print(f"Skipping {method}, model not found.")
                continue

        # 4) Plot the differences
        self.plot_attention_matrices(tokens, attention_matrices)

    def plot_attention_matrices(self, tokens, attention_matrices):
        """Plot attention weight differences between token pairs after fine-tuning."""
        methods = list(attention_matrices.keys())
        num_methods = len(methods)

        fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 6))
        if num_methods == 1:
            axes = [axes]

        # We'll unify the color scale
        vmin, vmax = 0, max(matrix.max() for matrix in attention_matrices.values())

        for ax, method in zip(axes, methods):
            sns.heatmap(
                attention_matrices[method],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="coolwarm",
                ax=ax,
                cbar=True,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(f"{method} vs Original MLM")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Tokens")

        plt.tight_layout()
        plt.show()

    def evaluate_finetuning(self, dataset, methods):
        """
        In this unsupervised scenario, `dataset` should be raw text lines,
        so no labels are needed. We'll store it for our Trainer.
        """
        self.dataset = dataset
        for method in methods:
            self.apply_finetuning(method)

    # Dummy collate_fn (Trainer will rely on DataCollatorForLanguageModeling)
    def collate_fn(self, batch):
        return batch


def run_mlm_finetuning():
    # 1) Load a dataset of raw text
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.load_edgar_dataset(ticker="AAPL", filing_type="10-K", amount=5)  # Should be unsupervised text only

    # 2) Initialize MLM FineTuner
    finetuner = MlmFineTuner(model_name="bert-base-uncased")

    # 3) Decide on your fine-tuning methods (PEFT or full)
    methods = ["mlm_full", "mlm_freeze_base", "mlm_bitfit", "mlm_lora"]

    # 4) Fine-tune with MLM
    finetuner.evaluate_finetuning(dataset, methods)

    # 5) Compare attention differences
    text = "Stocks soared on market optimism while bonds remained stable."
    finetuner.compare_attention_matrices(text, methods)


if __name__ == "__main__":
    run_mlm_finetuning()
