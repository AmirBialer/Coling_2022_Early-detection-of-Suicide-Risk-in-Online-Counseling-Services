from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, Subset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,AutoModel,AutoTokenizer, BertTokenizerFast, BertModel, pipeline,BertForMaskedLM

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score as fbeta
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import os
import time
import torch.nn.functional as F
import re
import datetime
import random

device = torch.device("cuda")
cross_entropy  = nn.NLLLoss()


def Generate_Input(df,tokenizer_name, tail_or_head,  max_len=512):
  if "pkl" in tokenizer_name:
    tokenizer=pickle.load(open(tokenizer_name,"rb"))
  else:
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

  if (tail_or_head=="keep_head"):
      tokenized = df.apply(lambda x: tokenizer.encode_plus(x, add_special_tokens=True,padding="max_length", max_length=max_len, return_attention_mask=True, truncation=True))#,,return_tensors ='pt' ))#tail ,
      input_ids=[el["input_ids"] for el in tokenized]
      attention_mask =[el["attention_mask"] for el in tokenized]
  elif tail_or_head=="keep_tail":
      tokenized = df.apply(lambda x: tokenizer.encode_plus(x, add_special_tokens=True,padding="max_length", max_length=max_len, return_attention_mask=True, truncation=False))
      input_ids=[el["input_ids"][-max_len:] for el in tokenized] 
      attention_mask =[el["attention_mask"][-max_len:] for el in tokenized]

  input_ids = torch.tensor(input_ids,dtype=torch.long)
  attention_mask = torch.tensor(attention_mask)
  return attention_mask, input_ids

def Choose_Label(dir_path,seed_val, batch_size=16,att_inp=(0,0)):
  seed_map={42:1,21:2,10:3,5:4,1:5}


  train_indices=pickle.load(open("/home/amirbial/Sahar/Data/Jan22/Train_Data_Splits/gsr_preprocessed_seeker_text_train_test_indices.pkl","rb"))
  g=pickle.load(open("/home/amirbial/Sahar/Data/Jan22/gsr_preprocessed_seeker_text_without_analyzes_samples.pkl","rb"))


  labels=pd.to_numeric(pd.Series([a["gsr"] for a in g]), downcast="integer")
  labels = pd.to_numeric(labels)

  train_in=train_indices[seed_map[seed_val]]["train"]
  test_in=train_indices[seed_map[seed_val]]["test"]


  if isinstance(att_inp[0],int):
    attention_mask=pickle.load(open(dir_path+"attention_mask.pkl", 'rb'))
    input_ids=pickle.load(open(dir_path+"input_ids.pkl", 'rb'))
  else:
    attention_mask,input_ids=att_inp


  labels=torch.tensor(labels,dtype=torch.long)
  dataset = TensorDataset(input_ids, attention_mask, labels)

  train_subset=Subset(dataset, train_in)
  test_subset=Subset(dataset, test_in)    
  train_dataloader = DataLoader(train_subset,  # The training samples.
              sampler = RandomSampler(train_subset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )
  test_dataloader = DataLoader(
              test_subset,  # The training samples.
              sampler = SequentialSampler(test_subset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )
  return train_subset, train_dataloader, test_subset, test_dataloader, ""

def Train_Test_Split(dataset,seed_val,batch_size=16, ratio=0.8):

  # Calculate the number of samples to include in each set.
  train_size = int(ratio * len(dataset))
  test_size = len(dataset) - train_size

  # Divide the dataset by randomly selecting samples.
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=torch.Generator().manual_seed(seed_val))
  train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
  test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = SequentialSampler(test_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
  return train_dataset, train_dataloader, test_dataset, test_dataloader

class BERT_Arch(nn.Module):
  def __init__(self, bert, num_labels):
    
    super(BERT_Arch, self).__init__()
    self.bert = bert 
    # dropout layer
    self.dropout = nn.Dropout(0.1)
    # dense layer 1
    self.DenseClassifier1 = nn.Linear(768,num_labels)

    #softmax activation function
    self.softmax = nn.LogSoftmax(dim=1)
    

  #define the forward pass
  def forward(self, sent_id, mask):
    #pass the inputs to the model  
    cls_hs=self.bert(sent_id,mask)[0][:,0,:]    #coreponds to cls token I believe
    #print("cls shape {}".format(cls_hs.shape))

    x=self.DenseClassifier1(cls_hs)
    # apply softmax activation
    x = self.dropout(x)

    x = self.softmax(x)
    return x


def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))
  
  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

  
def train_single_epoch(train_dataloader):
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  #for step,batch in enumerate(tqdm(train_dataloader)):
  for step,batch in enumerate(train_dataloader):  
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]

    sent_id, mask, labels = batch
    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)


    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    #logsoftmax activation relevant when using cross entropy
    #s=F.log_softmax(preds, dim=1) don't need because I changed to cross entropy from NLL
    #s=F.softmax(preds, dim=1)

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds

  
# function for evaluating the model
def evaluate(model, val_dataloader):
  
  print("\nEvaluating...")
  t0 = time.time()
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  predictions , true_labels = [], []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch



    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)
      preds_np = preds.detach().cpu().numpy()
      
      predictions.append(preds_np)
      true_labels.append(labels.detach().cpu().numpy())

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()
  
  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  flat_predictions_prob = np.array([np.exp(item) for sublist in predictions for item in sublist])#I added np.exp to convert logsoftmax to softmax
  #flat_predictions_prob = np.array([item for sublist in predictions for item in sublist])#I removed np.exp now that I use softmax
  flat_true_labels = [item for sublist in true_labels for item in sublist]
  return avg_loss, flat_predictions_prob,flat_true_labels

def Evaluate(model, val_dataloader):
  valid_loss, flat_predictions_prob,flat_true_labels= evaluate(model, val_dataloader)
  flat_predictions = np.argmax(flat_predictions_prob, axis=1).flatten()
  TN, FP, FN, TP = confusion_matrix(flat_true_labels, flat_predictions).ravel()
  sample_weight = compute_sample_weight(class_weight='balanced', y=flat_true_labels)
  results={"TP":TP, "TN":TN, "FP":FP, "FN":FN,"Accuracy":acu(flat_true_labels,flat_predictions),"Balanced Accuracy":bacu(flat_true_labels,flat_predictions,sample_weight=sample_weight), "ROC_AUC":auc(flat_true_labels,flat_predictions_prob[:,1]),"PR_AUC":pr_auc(flat_true_labels,flat_predictions_prob[:,1]),"Balanced PR_AUC":pr_auc(flat_true_labels,flat_predictions_prob[:,1],sample_weight=sample_weight),"Precision":precision_score(flat_true_labels,flat_predictions),"Balanced Precision":precision_score(flat_true_labels,flat_predictions,sample_weight=sample_weight), "Recall":recall_score(flat_true_labels,flat_predictions), "Balanced Recall":recall_score(flat_true_labels,flat_predictions,sample_weight=sample_weight),"F1_Pos":f1(flat_true_labels,flat_predictions, average="binary"), "F1_Weighted":f1(flat_true_labels,flat_predictions,sample_weight=sample_weight),"F1_micro":f1(flat_true_labels,flat_predictions, average="micro"), "F1_macro":f1(flat_true_labels,flat_predictions,average="macro"),"F2_Pos":fbeta(flat_true_labels,flat_predictions,average="binary",beta=2), "F2_Weighted":fbeta(flat_true_labels,flat_predictions,sample_weight=sample_weight, beta=2),"F2_micro":fbeta(flat_true_labels,flat_predictions, average="micro",beta=2), "F2_macro":fbeta(flat_true_labels,flat_predictions,average="macro",beta=2), "pred":flat_predictions,"prob":flat_predictions_prob}
  return results

def Train_Full_Cycle (train_dataloader,validation_dataloader,dir_path,label, epochs=5):
  t0 = time.time()
  # set initial loss to infinite
  best_valid_loss = float('inf')
  # empty lists to store training and validation loss of each epoch
  train_losses=[]
  valid_losses=[]
  val_roc_auc_list=[]
  val_acu_list=[]
  best_valid_roc_auc=0
  best_f1=0
  best_epoch=0
  #for each epoch
  for epoch in range(epochs):
      
      print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
      
      #train model
      train_loss, _ = train_single_epoch(train_dataloader)
      
      #evaluate model
      valid_loss, flat_predictions_prob,flat_true_labels= evaluate(model, validation_dataloader)
      flat_predictions = np.argmax(flat_predictions_prob, axis=1).flatten()
      
      # append training and validation loss
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      
      print(f'\nTraining Loss: {train_loss:.3f}')
      print(f'Validation Loss: {valid_loss:.3f}')
      # Calculate elapsed time in minutes.
      elapsed = format_time(time.time() - t0)
      print("elapsed time: {}".format(elapsed))
      print("predict "+str(np.unique(flat_predictions)))
      print("true "+str(np.unique(flat_true_labels)))
      results=Evaluate(model, val_dataloader=validation_dataloader)
      print('ROC-AUC: %.2f%%,  Accuracy: %.2f%%, Balanced Accuracy: %.2f%%, PR-AUC: %.2f%%, Balanced PR-AUC: %.2f%%'%(results["ROC_AUC"]*100,results["Accuracy"]*100,results["Balanced Accuracy"]*100, results["PR_AUC"]*100, results["Balanced PR_AUC"]*100))
      print('Precision: %.2f%%, Balanced Precision: %.2f%%, Recall: %.2f%%, Balanced Recall: %.2f%%' % (results["Precision"]*100, results["Balanced Precision"]*100, results["Recall"]*100, results["Balanced Recall"]*100))
      print('F1_Pos: %.2f%%, F1_Weighted: %.2f%%' % (results["F1_Pos"]*100, results["F1_Weighted"]*100))
      #save the best model
      val_roc_auc=results["ROC_AUC"]
      val_roc_auc_list.append(val_roc_auc)
      val_acu_list.append(results["Balanced Accuracy"])
      val_f1=results["F1_Pos"]
      if val_f1>=best_f1:
        best_epoch=epoch
        best_f1=val_f1
        #torch.save(model,dir_path+"Best_Model/trained_bert"+label+".pkl")
        save_pickle(model,dir_path+"Best_Model/trained_bert_pickled"+label+".pkl")
        save_pickle(results,dir_path+"Best_Model/results"+label+".pkl")
  print("best epoch: {}".format(best_epoch))
  return results, {"train_loss":train_losses, "val_loss":valid_losses, "val_roc_auc":val_roc_auc_list,"val_accuracy":val_acu_list}

def Train_80_20(train_dataset,bert,dir_path,seed_val,label, epochs, batch_size=16):

  train_dataset, train_dataloader, val_dataset, val_dataloader=Train_Test_Split(train_dataset,seed_val=seed_val,ratio=0.8)

  global model
  model = BERT_Arch(bert=bert, num_labels=2)
  for name, param in model.named_parameters(): #train just classification layer
    if ("cls" in name) or ("DenseClassifier1" in name) or ("classifier" in name.lower()):
      param.requires_grad=True
      print(name)
    else:
      param.requires_grad=False

  global optimizer
  optimizer = AdamW(model.parameters(),lr=5e-5)#, another option for lr 1e-3
  model = model.to(device)
  results,loss_dicts=Train_Full_Cycle(train_dataloader=train_dataloader,validation_dataloader=val_dataloader,dir_path=dir_path,label=label, epochs=epochs)
  return results, model,loss_dicts

def set_seed(seed_val):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

def create_dir_path(path,seed_val, tail_or_head,model_name):
  dir_path=path+str(seed_val)
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  dir_path+="/"+tail_or_head+"/"
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  dir_path+="/"+model_name+"/"
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  if not os.path.isdir(dir_path+"Best_Model"): os.makedirs(dir_path+"Best_Model")
  if not os.path.isdir(dir_path+"FT_Model"): os.makedirs(dir_path+"FT_Model")
  return dir_path

def save_pickle(file,filepath):
  p=re.findall(".*\/",filepath)[0]
  if not os.path.isdir(p): os.makedirs(p)
  pickle.dump(file, open(filepath,"wb"))


def Train_Cross_Validation(exp,label,text_series, tail_or_head,model_path,tokenizer_name,save_path, epochs):
  batch_size=32

  #load bert
  if model_path=="onlplab/alephbert-base"  or model_path=="avichr/heBERT":
    bert= transformers.BertModel.from_pretrained(model_path)#, trunc_medium=-1
  elif "pkl" in model_path:
    bert=pickle.load(open(model_path,"rb")).base_model
  else:
    bertconfig = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6, 
    type_vocab_size=1)
    bert_dict=torch.load(model_path+"/pytorch_model.bin")
    bert_lm=BertForMaskedLM(config=bertconfig)
    bert_lm.load_state_dict(state_dict=bert_dict)
    bert=bert_lm.base_model
    print("loaded from checkpoint")


  if tokenizer_name=="onlplab/alephbert-base" or tokenizer_name=="avichr/heBERT":
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
  else:
    tokenizer=pickle.load(open(tokenizer_name,"rb"))

  for seed_val in tqdm([1,5,10,21,42]):
    set_seed(seed_val=seed_val)
    dir_path=create_dir_path(path=save_path, seed_val=seed_val, tail_or_head=tail_or_head, model_path=model_path)


    attention_mask, input_ids=Generate_Input(text_series,tokenizer_name=tokenizer_name, tail_or_head=tail_or_head)
    save_pickle(attention_mask,dir_path+"attention_mask.pkl")
    save_pickle(input_ids,dir_path+"input_ids.pkl")

    train_dataset, train_dataloader, test_dataset, test_dataloader,__=Choose_Label(dir_path=dir_path,seed_val=seed_val,exp=exp,label=label,  tail_or_head=tail_or_head)

    print("start fold training")
    results,model, loss_dicts=Train_80_20(train_dataset=train_dataset, bert=bert,dir_path=dir_path,seed_val=seed_val,label=label, tail_or_head=tail_or_head, epochs=epochs,exp=exp, batch_size=batch_size)
    save_pickle(model,dir_path+"FT_Model/trained_bert_pickled"+label+".pkl")
    save_pickle(results,dir_path+"FT_Model/results"+label+".pkl")
    save_pickle(loss_dicts,dir_path+"FT_Model/loss_dicts"+label+".pkl")

    print("best trained model results")
    best_model=pickle.load(open(dir_path+"Best_Model/trained_bert_pickled"+label+".pkl","rb"))
    results=Evaluate(model=best_model,val_dataloader=test_dataloader)



    print(results)
    save_pickle(results,save_path+"_"+tail_or_head+"_"+str(seed_val)+"_"+label+".pkl")

    print('ROC-AUC: %.2f%%,  Accuracy: %.2f%%, Balanced Accuracy: %.2f%%, PR-AUC: %.2f%%, Balanced PR-AUC: %.2f%%'%(results["ROC_AUC"]*100,results["Accuracy"]*100,results["Balanced Accuracy"]*100, results["PR_AUC"]*100, results["Balanced PR_AUC"]*100))
    print('Precision: %.2f%%, Balanced Precision: %.2f%%, Recall: %.2f%%, Balanced Recall: %.2f%%' % (results["Precision"]*100, results["Balanced Precision"]*100, results["Recall"]*100, results["Balanced Recall"]*100))
    print('F1_Pos: %.2f%%, F1_Weighted: %.2f%%' % (results["F1_Pos"]*100, results["F1_Weighted"]*100))
    print("finish")

