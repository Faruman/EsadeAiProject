# Toxic comment classification
This project is based on the [kaggle toxic comments challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

A wrapper to process the data and create different models for this challenge was created within this repository. Hereby the different steps of the process are split into three different modules preprocessing, tokenization and model training which can be found in their respecitve files ([preprocessing.py](./preprocessing.py), [tokenization.py](./tokenization.py), [modeling.py](./modeling.py)).
Each of this modules allow to switch on and off different components:

```
preprocessing
 ├── doLower: transform input string to lower case
 ├── removeStopWords: remove stopwords defined by spacy nlp
 ├── doLemmatization: lemmatizes the words in the input string
 ├── doSpellingCorrection: corrects misspelled words
 └── removeNewLine: removes the \n character
```
```
tokenization
 └── tokenize_str: string which indicates which tokenizer to use (currently implemented are: BERTtokenizer, DistillBertTokenizer, RobertaTokenizer, fasttext, CountVectorizer, Tfidfvectorizer)
```
```
modeling
 ├── binaryClassification: defines if the model should split the problem into binary classifications or if it should use multilabel classification
 ├── labelSentences: if use binaryClassification with BERT-based models (Distilbert, Bert, Roberta), this dicts create the template for creating the artificial input samples
 ├── model_str: string defining which model to train
 ├── tokenizer: tokenizer for the model (needed for binaryClassification)
 ├── device: run on gpu or cpu
 ├── train_batchSize: size of the training batches
 ├── testval_batchSize: size of the test/ validation batches
 ├── learningRate: learning Rate
 ├── optimizer: which optimizer to use (currently only adam optimizer is implemented)
 ├── doLearningRateScheduler: activates the learning rate scheduler, decreasin the learningrate over time
 └── learningRateScheduler: option to supply a custom learning rate scheduler, default learning rate scheduler decreases the learning rate by factor ten each epoch
```

The evaluation of different model parameter combinations was done using the [wandb](https://wandb.ai) sweep functionality.

The results can be seen [here](https://wandb.ai/faruman/aiProject).