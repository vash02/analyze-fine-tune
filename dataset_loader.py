from datasets import load_dataset

class DatasetLoader:
    def __init__(self, path="financial_phrasebank", name="sentences_allagree"):
        self.path = path
        self.name = name
        self.dataset = load_dataset(path, name)
    def load_dataset(self):
        train_texts = self.dataset["train"]["sentence"]
        train_labels = self.dataset["train"]["label"]
        return [{"text": t, "label": l} for t, l in zip(train_texts, train_labels)]
