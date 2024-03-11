from model import Model
from torch import torch
from vncorenlp import VnCoreNLP
import os
import json
import random 
batch_size = 16
embedding_dim = 32
hidden_dim = 64
vocablength  = 280
# extract model, vocab in file

device = "cuda" if torch.cuda.is_available() else "cpu"
model_bot =Model(input_dim=vocablength,embedding_dim=embedding_dim,hidden_dim=hidden_dim ).to(device)

check_point = torch.load("weights\checkpoint3.pt", map_location=torch.device('cpu'))
word_to_idx = check_point["word_to_idx"]
idx_to_word = check_point["idx_to_word"]
model_bot.load_state_dict(check_point["state_dict"])

# ------------------------------------- Extracting response file --------------------------------------
with open("contentVn.json",encoding="utf-8") as f:
    data = json.load(f)
responeses = []
for intent in data["intents"]:
    responeses.append(intent["responses"])



segment = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
def wordsegmentation(sentence):
    text = segment.tokenize(sentence)
    text = ["<sos>"]+[ele for inlist in text for ele in inlist] +["<eos>"]
    return [word_to_idx.get(w,0) for w in text]



def answer_mesage(user_message):
    
    with torch.inference_mode():
        sample=torch.tensor(wordsegmentation(user_message))
        predict =model_bot(sample)
        predict_magnitude = predict.max().item()
        predict_value = predict.argmax().item()
        if predict_magnitude<0.7:
            return "Tôi không hiểu bạn đang nói gì "
        else:
            return random.choice(responeses[predict_value])
        
print(answer_mesage("alo alo"))