import spacy
import en_core_web_sm
import pandas as pd
from string import punctuation
punc = set(punctuation)
from tqdm import tqdm
tqdm.pandas()
from nltk import word_tokenize

import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


class Preprocessor():
    def __init__(self, doLower: bool, removeStopWords: bool, doLemmatization: bool, doSpellingCorrection: bool, removeNewLine: bool):
        self.doLower = doLower
        self.removeStopWords = removeStopWords
        self.doLemmatization = doLemmatization
        self.doSpellingCorrection = doSpellingCorrection
        self.removeNewLine = removeNewLine
        self.processor = None

    def fit(self, series: pd.Series):
        self.nlp = en_core_web_sm.load()
        def processor(text):
            if self.removeNewLine:
                text = text.replace("\n", "")
            if self.doLower:
                text = text.lower()
            if self.doSpellingCorrection:
                suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
                text = suggestions[0].term
            if self.doLemmatization:
                text_tokens = self.nlp(text)
                text_tokens = [token.lemma_ for token in text_tokens]
            else:
                text_tokens = word_tokenize(text)
            if self.removeStopWords:
                text_tokens = [word for word in text_tokens if not word in self.nlp.Defaults.stop_words]
            text_tokens = [x for x in text_tokens if x == " "]
            return ''.join(w if set(w) <= punc else ' '+w for w in text_tokens).lstrip()
        self.processor = processor

    def transform(self, series: pd.Series):
        return series.progress_apply(self.processor)

    def fit_transform(self, series: pd.Series):
        self.fit(series)
        return self.transform(series)