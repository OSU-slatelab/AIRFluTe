from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

# LSTM layer for pLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through pLSTM
# Note the input should have timestep%2 == 0
class pLSTMLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(pLSTMLayer, self).__init__()

        self.pLSTM = nn.LSTM(hidden_dim, hidden_dim, 1, bidirectional=False)
        self.linear1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_x, lens):
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        # reduce seq len by half
        lens = (lens / 2).long()
        # cut seq. len
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # pack sequence
        pack = pack_padded_sequence(input_x, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.pLSTM.flatten_parameters()
        output, hidden = self.pLSTM(pack)
        # pad packed seq output of LSTM
        out_lstm_pad, _ = pad_packed_sequence(output, batch_first=True)
        # Add and norm
        out = self.norm1(input_x + self.dropout(out_lstm_pad))
        # MLP
        out2 = self.linear2(self.dropout(F.relu(self.linear1(self.dropout(out)))))
        # Add and norm
        out = self.norm2(out + self.dropout(out2))
    
        return out, lens, hidden 


# Listener is a pLSTM stacking n layers to reduce time resolution 2^n times
class pLSTM(nn.Module):
    def __init__(self, hidden_dim, listener_layer, dropout=0.1):
        super(pLSTM, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        listener_hidden_dim = 2*hidden_dim
        self.pLSTM_layer0 = pLSTMLayer(listener_hidden_dim, dropout=dropout)
        dim = listener_hidden_dim
        for i in range(1,self.listener_layer):
            dim = 2*dim
            setattr(self, 'pLSTM_layer'+str(i), pLSTMLayer(dim, dropout=dropout))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens): # 32, seq, 80
        output, lens, _  = self.pLSTM_layer0(input_x, lens)
        for i in range(1,self.listener_layer):
            output, lens, _ = getattr(self,'pLSTM_layer'+str(i))(output, lens)
        return output, lens


class CustomLSTMLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(CustomLSTMLayer, self).__init__()

        self.pLSTM = nn.LSTM(hidden_dim, hidden_dim, 1, bidirectional=False)

        self.linear1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens):
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        # pack sequence
        pack = pack_padded_sequence(input_x, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.pLSTM.flatten_parameters()
        output, hidden = self.pLSTM(pack)
        # pad packed seq output of LSTM
        out_lstm_pad, _ = pad_packed_sequence(output, batch_first=True)
        # Add and norm
        out = self.norm1(input_x + self.dropout(out_lstm_pad))
        # MLP
        out2 = self.linear2(self.dropout(F.relu(self.linear1(self.dropout(out)))))
        # Add and norm
        out = self.norm2(out + self.dropout(out2))
    
        return out, lens, hidden 


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, nlayer, dropout=0.1):
        super(CustomLSTM, self).__init__()
        self.nlayer = nlayer
        self.LSTM_layer0 = CustomLSTMLayer(input_dim, dropout=dropout)
        for i in range(1,self.nlayer):
            setattr(self, 'LSTM_layer'+str(i), CustomLSTMLayer(input_dim, dropout=dropout))

    def forward(self, input, lens):
        output, lens, _  = self.LSTM_layer0(input, lens)
        for i in range(1,self.nlayer):
            output, lens, _ = getattr(self,'LSTM_layer'+str(i))(output, lens)
        return output, lens