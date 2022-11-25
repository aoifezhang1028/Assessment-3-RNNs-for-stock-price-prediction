import torch
import torch.nn as nn

# model section
# n is input size, 6 input stock indicators

class LSTM (nn.Module):
    def __init__(self, n):
        super(LSTM, self).__init__()
        self.rnn_layer = nn.LSTM(input_size=n, hidden_size=128, num_layers=2, batch_first=True)
        self.linear_layer = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        out1, (h_n, h_c) = self.rnn_layer(x)
        out2 = self.linear_layer(out1[:, -1, :])
        out2 = torch.flatten(out2)
        return out2


class GRU (nn.Module):
    def __init__(self, n):
        super(GRU, self).__init__()
        self.rnn_layer = nn.GRU(input_size=n, hidden_size=128, num_layers=2, batch_first=True)
        self.linear_layer = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        # difference to LSTM here not h_c (cell state)
        out1, h_n = self.rnn_layer(x)
        out2 = self.linear_layer(out1[:, -1, :])
        out2 = torch.flatten(out2)
        return out2


class CNNGRUModel(nn.Module):

    def __init__(self, n=6, lstm_units=64):
        super(CNNGRUModel, self).__init__()
        self.conv1d = nn.Conv1d(n, lstm_units, 1)
        self.act1 = nn.Tanh()
        self.drop = nn.Dropout(p=0.01)
        self.rnn_layer = nn.GRU(lstm_units, lstm_units, batch_first=True, num_layers=2, bidirectional=True)
        self.linear_layer = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):
        # important: CNN is applied in stock indicator [batch, date_period, stock indicator]
        # change the channel, make the stock indicator before data_period
        x = x.transpose(-1, -2)
        x = self.conv1d(x)
        x = self.act1(x)

        x = self.drop(x)

        # changing back
        x = x.transpose(-1, -2)
        out1, h_n = self.rnn_layer(x)
        out2 = self.linear_layer(out1[:, -1, :])
        out2 = torch.flatten(out2)
        return out2


# EMA class function
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# optimizer setup
def sgd_optimizer(model, lr, momentum, weight_decay):
    # bias not in weight decay
    optimizer = torch.optim.SGD([
        {'params': (p for name, p in model.named_parameters() if 'bias' not in name),
         'weight_decay': weight_decay},
        {'params': (p for name, p in model.named_parameters() if 'bias' in name)}
    ], lr=lr, momentum=momentum)

    return optimizer

