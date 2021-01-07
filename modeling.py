import wandb

import logging
import sys
import statistics
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch import optim

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, DistilBertForMultipleChoice, BertForMultipleChoice, RobertaForSequenceClassification, RobertaForMultipleChoice
from transformers import DistilBertTokenizer, BertTokenizer, RobertaTokenizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from nltk import word_tokenize

class Model():
    def __init__(self, binaryClassification: bool, model_str: str, doLower: bool, train_batchSize: int, testval_batchSize:int, learningRate: float, doLearningRateScheduler: bool, labelSentences: dict = None, max_label_len= None, model= None, optimizer= None, device= "cpu"):
        self.binaryClassification = binaryClassification
        self.labelSentences = labelSentences
        self.model_str = model_str
        self.tokenizer = None
        self.device = device
        self.train_batchSize = train_batchSize
        self.testval_batchSize = testval_batchSize
        self.learningRate = learningRate
        self.optimizer = optimizer
        self.doLearningRateScheduler = doLearningRateScheduler
        self.learningRateScheduler = None
        self.max_label_len = max_label_len

        if self.binaryClassification:
            self.num_labels = 1
        else:
            self.num_labels = len(self.labelSentences.keys())

        if self.model_str == "distilbert":
            if doLower:
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        elif self.model_str == "bert":
            if doLower:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif self.model_str == "roberta":
            self.model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
            self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        else:
            if model:
                if binaryClassification:
                    self.model = dict()
                    for key in self.labelSentences.keys():
                        self.model[key] = model
                else:
                    self.model = model
            else:
                logging.error("If model_str is not predefined, a model needs to be given.")
                sys.exit("If model_str is not predefined, a model needs to be given.")

    def preprocess(self, data: pd.Series, target, max_label_len):
        target = target.reset_index(drop=True)
        data = data.reset_index(drop=True)
        if self.model_str in ["distilbert", "bert", "roberta"]:
            df = pd.DataFrame([[a, b] for a, b in data.values], columns=["data", "mask"])
            df = pd.concat([df, target], axis=1)
            if self.binaryClassification:
                max_label_len += 2
                if type(self.labelSentences[list(self.labelSentences.keys())[0]]) == str:
                    for key in self.labelSentences.keys():
                        text = self.labelSentences[key]
                        temp = self.tokenizer(text, return_attention_mask=True, padding="max_length", truncation=True, max_length= max_label_len)
                        encoded_text = temp["input_ids"][1:]
                        mask = temp["attention_mask"][1:]
                        self.labelSentences[key] = (encoded_text, mask)
                max_label_len -= 1
                if set(target.columns).issubset(set(self.labelSentences.keys())):
                    def create_samples(df_row, target_columns):
                        base = [df_row["data"]]*len(target_columns)
                        mask = [df_row["mask"]]*len(target_columns)
                        for i, key in enumerate(target_columns):
                            extend_text, extend_mask = self.labelSentences[key]
                            last_data = np.max(np.nonzero(mask[i])) +1
                            if last_data < (len(mask[i])- len(extend_mask)):
                                base[i][last_data: (last_data+ len(extend_mask))] = extend_text
                                mask[i][last_data: (last_data+ len(extend_mask))] = extend_mask
                            else:
                                base[i][-(len(extend_text)+1):] = [base[i][last_data -1]] + extend_text
                                mask[i][-(len(extend_mask)+1):] = [base[i][last_data -1]] + extend_mask
                        df_row["data"] = np.array(base)
                        df_row["mask"] = np.array(mask)
                        return df_row
                    df = df.apply(create_samples, args= (target.columns,), axis=1)
                else:
                    logging.error("Target columns need to be subset of labelSentences.keys.")
                    sys.exit("Target columns need to be subset of labelSentences.keys.")
                return df["data"], df["mask"], target
            else:
                return df["data"], df["mask"], target
        else:
            mask = np.full(data.shape, 1)
            return data, mask, target

    def train(self, data, mask, target, device= "cpu"):
        data = torch.tensor(np.stack(data.values), dtype= torch.long)
        mask = torch.tensor(np.stack(mask.values), dtype= torch.long)
        target = torch.tensor(target.values, dtype= torch.int32)
        data = TensorDataset(data, mask, target)
        if self.binaryClassification:
            dataloader = DataLoader(data, batch_size=int(self.train_batchSize / len(self.labelSentences.keys())))
        else:
            dataloader = DataLoader(data, batch_size=self.train_batchSize)

        if self.model_str in ["distilbert", "bert", "roberta"]:
            self.model.train()
        else:
            if self.binaryClassification:
                for key in self.model:
                    self.model[key] = self.model[key].train()
                else:
                    self.model = self.model.train()

        if not self.optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), self.learningRate)
        if ~bool(self.learningRateScheduler) and self.doLearningRateScheduler:
            self.learningRateScheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            data, mask, target = batch

            self.optimizer.zero_grad()

            if self.binaryClassification:
                data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
                mask = mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2])
                target = target.reshape(target.shape[0]*target.shape[1])

                if self.model_str in ["distilbert", "bert", "roberta"]:
                    target = target.type(torch.float32)
                    model_output = self.model(input_ids= data, attention_mask= mask, labels= target)
                    loss = model_output.loss

                    loss.backward()
                    wandb.log({'train_batch_loss': loss.item()})

                    self.optimizer.step()
                else:
                    sum_loss = 0
                    for i, label in enumerate(self.target_columns):
                        model_output = self.model[label](input_ids=data, attention_mask=mask)
                        loss_fct = BCEWithLogitsLoss()
                        subtarget = target[:, i]
                        loss = loss_fct(model_output, subtarget)
                        sum_loss += loss.item()
                        loss.backward()

                    wandb.log({'train_batch_loss': sum_loss})

                    self.optimizer.step()
            else:
                if self.model_str in ["distilbert", "bert", "roberta"]:
                    model_output = self.model(input_ids= data, attention_mask= mask)
                    logits = model_output.logits

                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, target.type_as(logits))

                    loss.backward()
                    wandb.log({'train_batch_loss': loss.item()})

                    self.optimizer.step()

                else:
                    model_output = self.model(input_ids=data, attention_mask=mask)
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(model_output, target)

                    loss.backward()
                    wandb.log({'train_batch_loss': loss.item()})

                    self.optimizer.step()

    def test_validate(self, data, mask, target, type: str, device= "cpu"):
        data = torch.tensor(np.stack(data.values), dtype=torch.long)
        mask = torch.tensor(np.stack(mask.values), dtype=torch.long)
        target = torch.tensor(target.values, dtype=torch.int32)
        data = TensorDataset(data, mask, target)
        if self.binaryClassification:
            dataloader = DataLoader(data, batch_size= int(self.train_batchSize/len(self.labelSentences.keys())))
        else:
            dataloader = DataLoader(data, batch_size=self.train_batchSize)

        if self.model_str in ["distilbert", "bert", "roberta"]:
            self.model.eval()
        else:
            if self.binaryClassification:
                for key in self.model:
                    self.model[key] = self.model[key].eval()
                else:
                    self.model = self.model.eval()

        all_model_outputs= []
        all_targets = []

        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            data, mask, target = batch
            data = data.to(device)
            mask = mask.to(device)

            if self.binaryClassification:
                if self.model_str in ["distilbert", "bert", "roberta"]:
                    model_output = []
                    for i, label in enumerate(self.target_columns):
                        ind_model_output = self.model(input_ids=data[:, i, :], attention_mask=mask[:, i, :])
                        model_output.append(ind_model_output.logits)
                    model_output = torch.sigmoid(torch.cat(model_output, 1))

                else:
                    model_output = []
                    for i, label in enumerate(self.target_columns):
                        ind_model_output = self.model[label](input_ids=data, attention_mask=mask)
                        model_output.append(ind_model_output)
                    model_output = torch.sigmoid(torch.cat(model_output, 0))
            else:
                if self.model_str in ["distilbert", "bert", "roberta"]:
                    model_output = self.model(input_ids=data, attention_mask=mask)
                    model_output = torch.sigmoid(model_output.logits)

                else:
                    model_output = self.model(input_ids=data, attention_mask=mask)
                    model_output = torch.sigmoid(model_output)

            all_model_outputs.append(model_output.detach().cpu().numpy())
            all_targets.append(target)

        all_targets = np.concatenate(all_targets)
        all_model_outputs = np.concatenate(all_model_outputs)

        macroF1 = []
        macroPrec = []
        macroRec = []
        macroAuc = []
        for ind_target, ind_model_logits in zip(all_targets.transpose(), all_model_outputs.transpose()):
            macroF1.append(f1_score(ind_target, (ind_model_logits > 0.5).astype(int), average= "macro"))
            macroPrec.append(precision_score(ind_target, (ind_model_logits > 0.5).astype(int), average= "macro"))
            macroRec.append(recall_score(ind_target, (ind_model_logits > 0.5).astype(int), average= "macro"))
            try:
                macroAuc.append(roc_auc_score(ind_target, ind_model_logits, average="macro"))
            except:
                macroAuc.append(np.nan)
        macroF1 = statistics.mean(macroF1)
        macroPrec = statistics.mean(macroPrec)
        macroRec = statistics.mean(macroRec)
        macroAuc = statistics.mean(macroAuc)

        subsetAcc = accuracy_score(all_targets, (all_model_outputs > 0.5).astype(int))
        wandb.log({'{}_macroF1'.format(type): macroF1, '{}_macroPrec'.format(type): macroPrec, '{}_macroRec'.format(type): macroRec, '{}_subsetAcc'.format(type): subsetAcc, '{}_macroAuc'.format(type): macroAuc})

    def run(self, train_data, train_target, val_data, val_target, test_data, test_target, epochs: int):
        train_data, train_mask, train_target = self.preprocess(train_data, train_target, self.max_label_len)
        val_data, val_mask, val_target = self.preprocess(val_data, val_target, self.max_label_len)
        test_data, test_mask, test_target = self.preprocess(test_data, test_target, self.max_label_len)
        self.target_columns = list(train_target.columns)
        self.model.to(self.device)
        for i in range(epochs):
            print("epoch {}".format(i))
            self.train(train_data, train_mask, train_target, device= self.device)
            self.test_validate(val_data, val_mask, val_target, type= "validate", device= self.device)
            if self.learningRateScheduler:
                self.learningRateScheduler.step()
        self.test_validate(test_data, test_mask, test_target, type= "test", device= self.device)

    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)