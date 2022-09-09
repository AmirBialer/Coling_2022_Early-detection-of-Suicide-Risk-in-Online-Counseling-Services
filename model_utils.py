import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle, torch
from transformers import BertTokenizerFast
from torch.utils.data import Subset, random_split
import pandas as pd

class BERT_Arch(nn.Module):
    def __init__(self, bert, num_labels):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # dense layer 1
        self.DenseClassifier1 = nn.Linear(768, num_labels)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, mask)[0][:, 0, :]  # coreponds to cls token

        x = self.DenseClassifier1(cls_hs)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

def Generate_Input(df, tokenizer_name, tail_or_head, max_len=512):
    if "pkl" in tokenizer_name:
        tokenizer = pickle.load(open(tokenizer_name, "rb"))
    else:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    if (tail_or_head == "keep_head"):
        tokenized = df.apply(
            lambda x: tokenizer.encode_plus(x, add_special_tokens=True, padding="max_length", max_length=max_len,
                                            return_attention_mask=True,
                                            truncation=True))
        input_ids = [el["input_ids"] for el in tokenized]
        attention_mask = [el["attention_mask"] for el in tokenized]
    elif tail_or_head == "keep_tail":
        tokenized = df.apply(
            lambda x: tokenizer.encode_plus(x, add_special_tokens=True, padding="max_length", max_length=max_len,
                                            return_attention_mask=True, truncation=False))
        input_ids = [el["input_ids"][-max_len:] for el in tokenized]
        attention_mask = [el["attention_mask"][-max_len:] for el in tokenized]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask)
    return attention_mask, input_ids

def Choose_Label(train_indices_path, samples_path, dir_path, seed_val, batch_size=16, att_inp=(0, 0)):
    seed_map = {42: 1, 21: 2, 10: 3, 5: 4, 1: 5}

    train_indices = pickle.load(
        open(train_indices_path,"rb"))
    g = pickle.load(open(samples_path, "rb"))

    labels = pd.to_numeric(pd.Series([a["gsr"] for a in g]), downcast="integer")
    labels = pd.to_numeric(labels)

    train_in = train_indices[seed_map[seed_val]]["train"]
    test_in = train_indices[seed_map[seed_val]]["test"]

    if isinstance(att_inp[0], int):
        attention_mask = pickle.load(open(dir_path + "attention_mask.pkl", 'rb'))
        input_ids = pickle.load(open(dir_path + "input_ids.pkl", 'rb'))
    else:
        attention_mask, input_ids = att_inp

    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, labels)

    train_subset = Subset(dataset, train_in)
    test_subset = Subset(dataset, test_in)
    train_dataloader = DataLoader(train_subset,  # The training samples.
                                  sampler=RandomSampler(train_subset),  # Select batches randomly
                                  batch_size=batch_size  # Trains with this batch size.
                                  )
    test_dataloader = DataLoader(
        test_subset,  # The training samples.
        sampler=SequentialSampler(test_subset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return train_subset, train_dataloader, test_subset, test_dataloader, ""

def Train_Test_Split(dataset, seed_val, batch_size=16, ratio=0.8):

    # Calculate the number of samples to include in each set.
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(seed_val))
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    test_dataloader = DataLoader(
        test_dataset,  # The training samples.
        sampler=SequentialSampler(test_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return train_dataset, train_dataloader, test_dataset, test_dataloader
