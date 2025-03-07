import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorboardX
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy
from dataset_loader import DatasetLoader


class FineTuner:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(
            self.device)
        self.initial_weights = self.get_attention_weights()
        self.writer = tensorboardX.SummaryWriter(f'runs/FineTuning-{model_name}')
        self.step = 0

    def get_attention_weights(self):
        """ Get only attention weights before fine-tuning """
        return {name: param.clone().detach().cpu() for name, param in self.model.named_parameters() if
                "attention" in name}

    def extract_attention_scores(self, model, sentence):
        """ Extract attention scores for a given sentence """
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        return outputs.attentions

    def apply_finetuning(self, method="full"):
        model = deepcopy(self.model)
        model_path = f"./saved_models/{method}"

        if os.path.exists(model_path):
            print(f"Loading existing model for {method}")
            return AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        # Choose fine-tuning strategy
        if method == "full":
            for param in model.parameters():
                param.requires_grad = True
        elif method == "freeze_base":
            for name, param in model.named_parameters():
                if "distilbert" in name:
                    param.requires_grad = False
        elif method == "bitfit":
            for name, param in model.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False
        elif method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
            model = get_peft_model(model, peft_config).to(self.device)

        # Training setup
        training_args = TrainingArguments(
            output_dir=model_path,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            logging_dir="./logs",
            logging_steps=10,
            report_to="tensorboard"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.collate_fn
        )

        trainer.train()

        # Save the fine-tuned model
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        print(f"Fine-tuned model saved at: {model_path}")

        return model

    def compare_attention_weights(self, sentence, methods):
        """ Compare attention weights for a sentence across fine-tuning methods """
        attention_maps = {}

        for method in methods:
            model_path = f"./saved_models/{method}"
            if os.path.exists(model_path):
                model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
                attention_maps[method] = self.extract_attention_scores(model, sentence)
            else:
                print(f"Skipping {method}, model not found.")
                continue

        self.plot_attention_heatmaps(sentence, attention_maps)

    def plot_attention_heatmaps(self, sentence, attention_maps):
        """ Plot heatmaps of attention scores for tokens side by side """
        tokens = self.tokenizer.tokenize(sentence)
        num_methods = len(attention_maps)
        fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 6))

        if num_methods == 1:
            axes = [axes]

        for ax, (method, attentions) in zip(axes, attention_maps.items()):
            avg_attention = torch.mean(attentions[-1], dim=1).squeeze(0).cpu().numpy()
            sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, cmap="coolwarm", ax=ax, cbar=False)
            ax.set_title(f"{method}")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Tokens")

        plt.tight_layout()
        plt.show()

    def evaluate_finetuning(self, dataset, methods):
        self.dataset = dataset
        for method in methods:
            self.apply_finetuning(method)


def run_finetuning():
    dataset_load = DatasetLoader()
    dataset = dataset_load.load_dataset()
    finetuner = FineTuner(model_name="bert-base-uncased", num_labels=2)
    methods = ["full", "freeze_base", "bitfit", "lora"]
    finetuner.evaluate_finetuning(dataset, methods)

    sentence = "The quick brown fox jumps over the lazy dog."
    finetuner.compare_attention_weights(sentence, methods)


if __name__ == "__main__":
    run_finetuning()
