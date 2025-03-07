import matplotlib.pyplot as plt
import seaborn as sns
import torch


class AttentionVisualizer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_attention_weights(self, text):
        """Extracts attention weights for a given text input."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=20).to(
            self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        return outputs.attentions  # List of attention layers (12 layers for BERT-base)

    def visualize_attention(self, text, method, layer=0, head=0):
        """
        Visualizes the attention weight distribution for a specific layer and head.
        :param text: Input sentence
        :param method: Fine-tuning method name (for labeling)
        :param layer: Transformer layer to visualize (default = 0)
        :param head: Attention head to visualize (default = 0)
        """
        attentions = self.get_attention_weights(text)
        if attentions is None:
            print("No attention weights found!")
            return

        attention_matrix = attentions[layer][0, head].cpu().numpy()  # Shape: (seq_len, seq_len)
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))[:attention_matrix.shape[0]]

        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)
        plt.title(f"Attention Heatmap | {method} | Layer {layer}, Head {head}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=90)
        plt.show()
