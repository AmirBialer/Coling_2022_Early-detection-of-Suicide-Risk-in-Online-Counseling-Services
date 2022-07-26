from xml.sax.handler import feature_external_ges
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, Subset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import transformers
from transformers import AdamW, BertConfig,AutoModel,AutoTokenizer, BertTokenizerFast, BertModel, pipeline
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pickle
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score as bacu
from sklearn.metrics import accuracy_score as acu
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score as pr_auc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import fbeta_score as fbeta

from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import os
import time
import torch.nn.functional as F
import datetime
import random
import sys
import re
sys.path.insert(1, "/home/amirbial/Sahar/Code/")
from BERT_MODEL import *

now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
parser.add_argument("--tail_or_head")
parser.add_argument("--epochs")
parser.add_argument("--label")
parser.add_argument("--exp")


args = parser.parse_args()
epochs=int(args.epochs)
print("epochs: "+str(epochs))
tail_or_head=str(args.tail_or_head)
print("tail_or_head: "+str(tail_or_head))
model_path=str(args.model_path)
print("model_path :"+model_path)
label=str(args.label)
print("label :"+label)
exp=str(args.exp)
print("exp :"+exp)



Train_Cross_Validation(exp=exp,label=label,seeker_vol_all="seeker",  tail_or_head=tail_or_head,model_path=model_path, epochs=epochs)