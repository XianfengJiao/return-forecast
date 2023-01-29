import torch.nn as nn

class MLP_resnet(nn.Module):
    def __init__(self, n_input=64, n_feature=4, hidden_dim=32, output_dim=1, dropout_rate=0.5):
        super().__init__()
        
        self.fc_feature = nn.Linear(n_feature, 1)
        self.fc1 = nn.Linear(n_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, input): # input = [batch_size, stock_num(10), 64, 4] 
        x = self.fc_feature(input=input).squeeze() # middle_output = [batch_size, stock_num(10), 64] 
        
        x = self.dropout(x)

        x = self.fc1(x) # batch_size x feature_size
        x = self.relu(x)
        x = self.dropout(x)

        x = x + self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x + self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x + self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        o = self.fc5(x).squeeze()
        
        return o