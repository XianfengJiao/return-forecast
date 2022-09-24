import torch
import math
from torch import nn, Tensor

class GRU(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=64, output_dim=1, num_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        self.gru = torch.nn.GRU(feature_dim, hidden_dim, batch_first = True)
        self.l_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, input):
        _, stock_num, _, _ = input.size()
        output_arr = []
        for i in range(stock_num):
            stock_input = input[:,i,:,:]
            _, hidden_t = self.gru(stock_input)
            hn = hidden_t[0][-1].squeeze()
            hn = hidden_t.squeeze()
            o = self.l_out(hn)
            output_arr.append(o)
        output = torch.cat(output_arr,dim=-1)

        return output
