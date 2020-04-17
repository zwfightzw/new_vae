import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            # w.data.uniform_(-std, std)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  # Change nonlinearity to 'leaky_relu' if you switch
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden, w=None):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        if w is None:
            inputgate = F.sigmoid(i_i + h_i)
        else:
            inputgate = F.sigmoid(i_i + h_i) * w
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        #std = 1.0 / math.sqrt(self.hidden_size)
        for m in self.modules():
            #w.data.uniform_(-std, std)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                nonlinearity='relu')  # Change nonlinearity to 'leaky_relu' if you switch
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        #gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return hy, cy