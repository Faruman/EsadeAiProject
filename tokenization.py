import pandas as pd
from transformers import DistilBertTokenizer, BertTokenizer, RobertaTokenizer
from nltk import word_tokenize
import fasttext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm
tqdm.pandas()

class Tokenizer():
    def __init__(self, tokenizeStr: str, fasttextFile: str, doLower: bool):
        self.tokenize_str = tokenizeStr
        self.fasttextFile = fasttextFile
        self.doLower = doLower
        self.tokenizer = None

    def fit(self, series: pd.Series):
        if self.tokenize_str == "bert":
            if self.doLower:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            def generate_BERT_vectors(s):
                toks = tokenizer(s,  return_attention_mask= True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_BERT_vectors
        elif self.tokenize_str == "distilbert":
            if self.doLower:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            def generate_DistilBERT_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_DistilBERT_vectors
        elif self.tokenize_str == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
            def generate_RoBERTa_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_RoBERTa_vectors
        elif self.tokenize_str == "fasttext":
            embeddingModel = fasttext.load_model(self.fasttextFile)
            def generate_fasttext_vectors(s):
                words = word_tokenize(s)
                words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                return words_embed
            self.tokenizer = generate_fasttext_vectors
        elif self.tokenize_str == "bow":
            vectorizer = CountVectorizer()
            vectorizer.fit(series)
            self.tokenizer = vectorizer.transform
        elif self.tokenize_str == "tfidf":
            vectorizer = TfidfVectorizer()
            vectorizer.fit(series)
            self.tokenizer = vectorizer.transform

    def transform(self, series: pd.Series):
        return series.progress_apply(self.tokenizer)

    def fit_transform(self, series: pd.Series):
        self.fit(series)
        return self.transform(series)