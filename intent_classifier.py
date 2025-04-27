from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from optimum.intel import INCModelForSequenceClassification
import torch

class QuantizedIntentClassifier:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = INCModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            load_in_8bit=True,
            device_map="auto"
        )
        self.intents = ["search_product", "price_inquiry", "filter_products"]

    def get_top_intent(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.intents[torch.argmax(outputs.logits).item()]
