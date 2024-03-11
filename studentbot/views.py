from django.shortcuts import render
from django.views import generic
from django.http.response import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pprint import pprint
import requests
# Create your views here.
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
        if predict_magnitude<0.5:
            return "Tôi không hiểu bạn đang nói gì "
        else:
            return random.choice(responeses[predict_value])
        

@csrf_exempt
def studentresponse(request):
    if request.method == 'GET':
        token = request.GET['hub.verify_token']
        challenge = request.GET['hub.challenge']
        if token == 'Bi2705':
            return HttpResponse(challenge)
        else:
            return HttpResponse('Error, invalid verification token')
    elif request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        pprint(data)
        for entry in data['entry']:
            for messaging_event in entry['messaging']:
                if messaging_event.get('message',0):
                    sender_id = messaging_event['sender']['id']
                    recipient_id = messaging_event['recipient']['id']
                    message_text = messaging_event['message']['text']
                    bot_text = answer_mesage(message_text)
                    pprint(message_text)
                    pprint(bot_text)
                    # pprint(sender_id)
                    send_message(sender_id,bot_text)
                    # pprint(answer_mesage(message_text))
        return HttpResponse()
    # return HttpResponse("HelloWorld")


def send_message(recipient_id, message_text):
    params = {
        'access_token': "EAAK5ilyzBZAIBOw3DeBvvJfRUrQWeqRllRo9Qm1f6ktS1FD9EoorTotXLFx0atkZBNa8AoxZBbGenPd8l6KuJEJGQQ56z6QdE5Mg0QSiEX67CAym35Xk3BcMEsQhE1ZBHzxrwnJxl5B6iGdyInG3n2kHcODabPpurOpFm8kGa0jvSWlFBCZAGqZA6rkMnG",
        'recipient': json.dumps({'id': recipient_id}),
        'message': json.dumps({'text': message_text})
    }
    headers = {'Content-type': 'application/json'}
    r = requests.post('https://graph.facebook.com/v12.0/me/messages', params=params, headers=headers)

