import torch
from transformers import BartTokenizer, BartForConditionalGeneration
 
class NewsSummariser:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
 
    def summarise(self, text, max_length=130, min_length=30):
        inputs = self.tokenizer(text, return_tensors='pt',
                                max_length=1024, truncation=True)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            num_beams=4, max_length=max_length,
            min_length=min_length, length_penalty=2.0,
            early_stopping=True, no_repeat_ngram_size=3)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
    def extractive_fallback(self, text, n_sentences=3):
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)
        words = text.lower().split()
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        scores = []
        for s in sentences:
            score = sum(freq.get(w.lower(), 0) for w in s.split())
            scores.append((score, s))
        top = sorted(scores, reverse=True)[:n_sentences]
        return ' '.join(s for _, s in sorted(top, key=lambda x: text.index(x[1])))
 
article = """Scientists have developed a new machine learning model that can predict
protein structures with unprecedented accuracy. The model, trained on millions of
protein sequences, uses a transformer architecture similar to those used in natural
language processing. This breakthrough could accelerate drug discovery and help
researchers understand diseases at the molecular level. The team published their
findings in Nature and made the code publicly available."""
 
summariser = NewsSummariser.__new__(NewsSummariser)
summariser.tokenizer = None; summariser.model = None
print("Extractive summary:")
print(summariser.extractive_fallback(article, 2))
print("\n[BART abstractive summary requires: pip install transformers torch]")
