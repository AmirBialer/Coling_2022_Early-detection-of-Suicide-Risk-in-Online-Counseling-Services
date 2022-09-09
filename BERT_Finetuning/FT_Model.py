import sys
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
epochs = int(args.epochs)
print("epochs: " + str(epochs))
tail_or_head = str(args.tail_or_head)
print("tail_or_head: " + str(tail_or_head))
model_path = str(args.model_path)
print("model_path :" + model_path)
label = str(args.label)
print("label :" + label)
exp = str(args.exp)
print("exp :" + exp)

Train_Cross_Validation(exp=exp, label=label, seeker_vol_all="seeker", tail_or_head=tail_or_head, model_path=model_path,
                       epochs=epochs)
