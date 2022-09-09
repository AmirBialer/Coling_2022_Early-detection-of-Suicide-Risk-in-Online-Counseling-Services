import psutil
import humanize
import GPUtil as GPU
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import BertConfig
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datetime import datetime
import pickle
import torch
import argparse
import os


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

GPU.showUtilization()
GPUs = GPU.getGPUs()
print (GPUs[0])
gpu = GPUs[0]
def printm(txt):
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    print("***********************")
    print("Memory Stats - " + txt)
    print("***********************")
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    print("***********************")
printm("Program Startup")
random_seed=int(42)
print("random seed: "+str(42))

parser = argparse.ArgumentParser()
parser.add_argument("--sentences_path")
parser.add_argument("--dir_path")
args = parser.parse_args()
sentences_path=str(args.sentences_path)
dir_path=str(args.dir_path)


if not os.path.isdir(dir_path): os.makedirs(dir_path)
if not os.path.isdir(dir_path+"/Tokenizer"): os.makedirs(dir_path+"/Tokenizer")
if not os.path.isdir(dir_path+"Final_Model"): os.makedirs(dir_path+"Final_Model")
if not os.path.isdir(dir_path+"Final_Model/Tokenizer"): os.makedirs(dir_path+"Final_Model/Tokenizer")



# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=sentences_path, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.save_model(dir_path+"/Tokenizer")
print(torch.cuda.is_available())


bertconfig = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = BertTokenizerFast.from_pretrained(dir_path+"./Tokenizer", max_len=512)

model = BertForMaskedLM(config=bertconfig)

print(model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= sentences_path,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir=dir_path,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    save_strategy ="steps",
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


trainer.train()
trainer.save_model(dir_path)
model.save_pretrained(dir_path+"Final_Model/")
tokenizer.save_pretrained(dir_path+"Final_Model/Tokenizer")
pickle.dump(model,open(dir_path+"Final_Model/Final_Model.pkl","wb"))
pickle.dump(tokenizer,open(dir_path+"Final_Model/Tokenizer/Tokenizer.pkl","wb"))
print("done")
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

