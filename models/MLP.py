import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_input=64, n_feature=4, n_1=32, n_2=16, n_3=4, n_4=1, dropout_rate=0.5):
        super().__init__()
        
        self.fc_feature = nn.Linear(n_feature, 1)
        self.linear1 = nn.Linear(n_input, n_1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_1, n_2)
        self.linear3 = nn.Linear(n_2, n_3)
        self.linear4 = nn.Linear(n_3, n_4)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, input): # input = [batch_size, stock_num(10), 64, 4] 
        middle_output = self.fc_feature(input=input).squeeze() # middle_output = [batch_size, stock_num(10), 64] 
        middle_output = self.dropout(middle_output)
        output_ = self.linear2(self.relu(self.linear1(middle_output)))
        output = self.linear4(self.relu(self.linear3(output_))).squeeze()
        
        return output