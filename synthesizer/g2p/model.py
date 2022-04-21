import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, graphemes_size, hidden_size):
        super(Encoder, self).__init__()
        self.graphemes_size = graphemes_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(graphemes_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, x):
        # x: TxN
        T, N = x.size()
        emb = self.emb(x)  # emb: TxNxH
        output, _ = self.rnn(emb)  # output: TxNxH, hidden: 1XNxH
        output = output.view(T, N, 2, -1).sum(2)  # reduce bi-RNN by sum
        return output


class Decoder(nn.Module):
    def __init__(self, phonemes_size, hidden_size):
        super(Decoder, self).__init__()
        self.phonemes_size = phonemes_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(phonemes_size, hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, phonemes_size, bias=False)

    def forward(self, x, enc, hidden):
        # x: 1xN, enc: T(enc)xNxH, hidden: 1xNxH
        emb = self.emb(x)  # emb: 1xNxH

        shaped_hidden = hidden.squeeze(0).unsqueeze(2).contiguous()
        shaped_act_hidden = torch.tanh(shaped_hidden)
        shaped_enc = enc.transpose(0, 1).contiguous()
        shaped_act_enc = torch.tanh(shaped_enc)
        e = torch.bmm(shaped_act_enc, shaped_act_hidden)  # NxT(enc)x1
        att_weights = torch.softmax(e, 1).squeeze(2).unsqueeze(1)
        att_vec = torch.bmm(att_weights, shaped_enc)
        att_vec = att_vec.transpose(1, 0).contiguous()

        x = torch.cat([att_vec, emb], dim=2)
        x = self.attn_combine(x)

        output, hidden = self.rnn(x, hidden)  # output: 1xNxH, hidden: 1XNxH

        T, N, H = output.size()
        output = output.view(T * N, H)
        output = self.fc(output)
        output = output.view(T, N, self.phonemes_size)

        return output, hidden, att_weights
