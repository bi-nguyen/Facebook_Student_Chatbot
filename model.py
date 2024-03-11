import torch.nn as nn
import torch 
class Model(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,output_dim=13,p=0.5 ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_dim,batch_first=True)
        self.linear = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        out = self.dropout(self.embedding(x))
        out,h = self.rnn(out)
        out = self.linear(h.squeeze(0))
        return out