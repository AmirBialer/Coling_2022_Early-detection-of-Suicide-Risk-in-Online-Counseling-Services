import psutil
import humanize
import GPUtil as GPU
from transformers import BertConfig
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datetime import datetime
import pickle
import torch
import argparse
import os
import random
import numpy as np

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

GPU.showUtilization()
GPUs = GPU.getGPUs()
print(GPUs[0])
gpu = GPUs[0]


def printm(txt):
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    print("***********************")
    print("Memory Stats - " + txt)
    print("***********************")
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree,
                                                                                                    gpu.memoryUsed,
                                                                                                    gpu.memoryUtil * 100,
                                                                                                    gpu.memoryTotal))
    print("***********************")


printm("Program Startup")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs")
parser.add_argument("--model_name")
parser.add_argument("--model_path")
parser.add_argument("--tokenizer_path")
parser.add_argument("--sentences_path")
parser.add_argument("--dir_path")

args = parser.parse_args()
epochs = float(args.epochs)
model_name = str(args.model_name)
model_path = str(args.model_path)
tokenizer_path = str(args.tokenizer_path)
sentences_path = str(args.sentences_path)
dir_path = str(args.dir_path)

print("epochs: " + str(epochs))
print("model_name: " + str(model_name))
print("model_path: " + str(model_path))
print("tokenizer_path: " + str(tokenizer_path))
device = torch.device("cuda")
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
print(torch.cuda.is_available())

bertconfig = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

model = pickle.load(open(model_path, "rb"))
tokenizer = pickle.load(open(tokenizer_path, "rb"))

print(model.num_parameters())

model.to(device)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=sentences_path,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
training_args = TrainingArguments(
    output_dir=dir_path,
    overwrite_output_dir=True,
    num_train_epochs=epochs,  # for debugging=0.0001
    per_device_train_batch_size=32,
    save_steps=10_000,
    logging_strategy="epoch",
    save_total_limit=2,
    max_steps=-1,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

pickle.dump(tokenizer, open(dir_path + str(epochs) + "epochs_Tokenizer.pkl", "wb"))
log = trainer.train()
pickle.dump(model, open(dir_path + str(epochs) + "epochs_Model.pkl", "wb"))
print(log)
pickle.dump(log, open(dir_path + str(epochs) + "epochs_log.pkl", "wb"))

print("done!")
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
