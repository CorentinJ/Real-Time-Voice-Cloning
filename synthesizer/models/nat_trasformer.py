from math import sqrt
import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
from torch.nn import functional as F
from synthesizer.layers import ConvNorm, LinearNorm
from synthesizer.utils import to_gpu, get_mask_from_lengths
import sys
import math
import os
from synthesizer.hparams import hparams
np.set_printoptions(sys.maxsize)


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead=4, nhid=1024, nlayers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        if ninp % 2 == 0:
            self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, src_mask=None):
        src = src * sqrt(self.ninp)
        if self.ninp % 2 == 0:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.num_mels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int(
                                 (hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.num_mels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.num_mels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            hparams.encoder_embedding_dim, 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, speaker_embedding):
        for conv in self.convolutions:
            x = F.dropout(conv(x), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = F.dropout(outputs, hparams.p_decoder_dropout, self.training)

        if speaker_embedding is not None:
            outputs = self.add_speaker_embedding(outputs, speaker_embedding)

        return outputs

    def inference(self, x, speaker_embedding):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if speaker_embedding is not None:
            outputs = self.add_speaker_embedding(outputs, speaker_embedding)

        return outputs

    def add_speaker_embedding(self, x, speaker_embedding):
        # SV2TTS
        # The input x is the encoder output and is a 3D tensor with size (batch_size, num_chars, tts_embed_dims)
        # When training, speaker_embedding is also a 2D tensor with size (batch_size, speaker_embedding_size)
        #     (for inference, speaker_embedding is a 1D tensor with size (speaker_embedding_size))
        # This concats the speaker embedding for each char in the encoder output

        # Save the dimensions as human-readable names
        batch_size = x.size()[0]
        num_chars = x.size()[1]

        if speaker_embedding.dim() == 1:
            idx = 0
        else:
            idx = 1

        # Start by making a copy of each speaker embedding to match the input text length
        # The output of this has size (batch_size, num_chars * tts_embed_dims)
        speaker_embedding_size = speaker_embedding.size()[idx]
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)

        # Reshape it and transpose
        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)

        # Concatenate the tiled speaker embedding with the encoder output
        x = torch.cat((x, e), 2)
        return x


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.num_mels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.speaker_embedding_size = hparams.speaker_embedding_size

        self.prenet = Prenet(
            hparams.num_mels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_lstm = nn.LSTMCell(
            hparams.prenet_dim + 2 * hparams.encoder_embedding_dim +
            hparams.speaker_embedding_size + hparams.pos_enc_dim,
            hparams.decoder_lstm_dim)

        self.decoder_lstm = nn.LSTMCell(
            hparams.decoder_lstm_dim, hparams.decoder_lstm_dim)

        self.linear_projection = LinearNorm(
            hparams.decoder_lstm_dim + 2*hparams.encoder_embedding_dim +
            hparams.speaker_embedding_size,
            hparams.num_mels)

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """

        B = memory.size(0)

        self.attention_hidden = Variable(memory.data.new(
            B, hparams.decoder_lstm_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, hparams.decoder_lstm_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, hparams.decoder_lstm_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, hparams.decoder_lstm_dim).zero_())

    def parse_decoder_outputs(self, mel_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:

        RETURNS
        -------
        mel_outputs:
        """
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs

    def decode(self, decoder_input, sampler_output):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        sample_output: previous sample output

        RETURNS
        -------
        mel_output:
        """

        self.attention_hidden, self.attention_cell = self.attention_lstm(
            decoder_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        self.decoder_hidden, self.decoder_cell = self.decoder_lstm(
            self.attention_hidden, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        # print(self.decoder_hidden.size())
        # print(self.decoder_hidden.size(), sampler_output.size())
        proj_input = torch.cat(
            (self.decoder_hidden, sampler_output), 1)  # [B, 1024 + 1280]

        decoder_output = self.linear_projection(proj_input)

        return decoder_output

    def forward(self, memory, sampler_outputs, decoder_inputs):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Upsample outputs + Positional Embedding
        sampler_outputs: Upsample outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        # print(decoder_inputs.size())
        # decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        sampler_outputs = sampler_outputs.transpose(0, 1)  # [T, B, 1280]
        decoder_inputs = decoder_inputs.transpose(0, 1)  # [T, B , Mel]
        decoder_inputs = self.prenet(decoder_inputs)  # [T, B, 256]

        # [T, B, 1312 + 256]
        decoder_inputs = torch.cat((decoder_inputs, memory.transpose(0, 1)), 2)

        self.initialize_decoder_states(memory)

        mel_outputs = list()

        while len(mel_outputs) < decoder_inputs.size(0):
            decoder_input = decoder_inputs[len(mel_outputs)]
            sample_output = sampler_outputs[len(mel_outputs)]
            mel_output = self.decode(decoder_input, sample_output)
            # print(mel_output.size())
            mel_outputs += [mel_output.squeeze(1)]
            #     gate_outputs += [gate_output.squeeze(1)]
            #     alignments += [attention_weights]

        mel_outputs = self.parse_decoder_outputs(mel_outputs)

        return mel_outputs

    def get_initial_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels).zero_())
        # print(decoder_input.size())
        return decoder_input

    def inference(self, memory, sampler_outputs):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs [B, T, 1312]

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_initial_frame(memory)  # [B, Mel]
        sampler_outputs = sampler_outputs.transpose(0, 1)  # [T, B, 1280]
        self.initialize_decoder_states(memory)

        mel_outputs = list()
        while len(mel_outputs) < sampler_outputs.size(0):
            decoder_input = self.prenet(decoder_input)
            decoder_inputs = torch.cat(
                (decoder_input, memory.transpose(0, 1)[len(mel_outputs)]), 1)
            sample_output = sampler_outputs[len(mel_outputs)]
            mel_output = self.decode(decoder_inputs, sample_output)
            mel_outputs += [mel_output.squeeze(1)]
            decoder_input = mel_output

        mel_outputs = self.parse_decoder_outputs(mel_outputs)
        return mel_outputs


class DurationPredictor(nn.Module):
    def __init__(self, hparams):
        super(DurationPredictor, self).__init__()
        self.transformer = TransformerModel(
            hparams.encoder_embedding_dim * 2 + hparams.speaker_embedding_size)
        # self.lstm1 = nn.LSTM(hparams.encoder_embedding_dim * 2 + hparams.speaker_embedding_size,
        #                      hparams.encoder_embedding_dim, 1,
        #                      batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(hparams.encoder_embedding_dim * 2,
        #                      hparams.encoder_embedding_dim, 1,
        #                      batch_first=True, bidirectional=True)
        self.projection = nn.Linear(
            hparams.encoder_embedding_dim*2 + hparams.speaker_embedding_size, 1)
        # self.projection = nn.Linear(
        #     hparams.encoder_embedding_dim*2, 1)

    def forward(self, x):  # [B, L, 1280]
        outputs = self.transformer(x)
        # print(outputs.size())

        # self.lstm1.flatten_parameters()
        # outputs = self.transform(x,)
        # outputs, _ = self.lstm1(x)
        # outputs = F.dropout(outputs, hparams.p_decoder_dropout, self.training)
        # # self.lstm2.flatten_parameters()
        # outputs, _ = self.lstm2(outputs)
        # outputs = F.dropout(outputs, hparams.p_decoder_dropout, self.training)
        x = self.projection(outputs)
        # return x


class RangePredictor(nn.Module):
    def __init__(self, hparams):
        super(RangePredictor, self).__init__()
        # self.transformer = TransformerModel(
        #     hparams.encoder_embedding_dim * 2 + hparams.speaker_embedding_size + 1)
        self.lstm1 = nn.LSTM(hparams.encoder_embedding_dim * 2 + hparams.speaker_embedding_size + 1,
                             hparams.encoder_embedding_dim, 1,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hparams.encoder_embedding_dim * 2,
                             hparams.encoder_embedding_dim, 1,
                             batch_first=True, bidirectional=True)
        self.projection = nn.Linear(
            hparams.encoder_embedding_dim*2, 1)
        # self.projection = nn.Linear(
        #     hparams.encoder_embedding_dim*2 + hparams.speaker_embedding_size + 1, 1)

    def forward(self, x):
        # self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x)
        outputs = F.dropout(outputs, hparams.p_decoder_dropout, self.training)
        # self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)
        outputs = F.dropout(outputs, hparams.p_decoder_dropout, self.training)
        # outputs = self.transformer(x)
        x = F.softplus(self.projection(outputs))
        return x


class GaussianUpsample(nn.Module):
    def __init__(self, hparams):
        super(GaussianUpsample, self).__init__()
        pass

    def forward(self, encoder_outputs, durations, range_outputs, device='cuda'):
        x = torch.sum(durations, dim=-1, keepdim=True)  # [B, 1]
        e = torch.cumsum(durations, dim=-1).float()  # [B,  L]
        c = (e - 0.5 * durations).unsqueeze(-1)  # [B, L, 1]
        t = torch.arange(0, torch.max(
            x)).unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, T]
        w_1 = torch.exp(-(range_outputs**-2) * ((t-c) ** 2))
        w_2 = torch.sum(torch.exp(-(range_outputs**-2) *
                                  ((t-c) ** 2)), dim=1, keepdim=True)
        w = w_1/w_2  # [B, L, T]
        out = torch.matmul(w.transpose(
            1, 2), encoder_outputs)  # [B, T, ENC_DIM]
        # out = w.transpose(1, 2) @ encoder_outputs

        alignment = torch.zeros(durations.size(0), torch.max(x).item())
        alignment = self.create_alignment(
            alignment, durations.cpu().numpy()).to(device)

        return alignment.unsqueeze(-1), out

    def create_alignment(self, base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape

        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count+k] = k+1
                count = count + duration_predictor_output[i][j]
        return base_mat


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NonAttentiveTacotron(nn.Module):
    def __init__(self, hparams):
        super(NonAttentiveTacotron, self).__init__()
        self.mask_padding = hparams.mask_padding
        # self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.num_mels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.tts_embed_dims))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.duration_predictor = DurationPredictor(hparams)
        self.range_predictor = RangePredictor(hparams)
        self.gaussian_upsampling = GaussianUpsample(hparams)
        self.pos_enc = PositionalEncoding(hparams.pos_enc_dim)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def parse_batch(self, batch):

        text = torch.from_numpy(batch["text"]).long().to(self.device)
        mel_target = torch.from_numpy(
            batch["mel_target"]).float().to(self.device)
        durations = torch.from_numpy(batch["durations"]).long().to(self.device)
        embeds = torch.from_numpy(batch["embeds"]).long().to(self.device)
        src_len = torch.from_numpy(batch["src_len"]).long().to(self.device)
        mel_len = torch.from_numpy(batch["mel_len"]).long().to(self.device)
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)

        return ((text, src_len, mel_target, mel_len, durations, max_src_len, max_mel_len, embeds), (mel_target, mel_len))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        self.step += 1
        text, src_len, mel_target, mel_len, durations, max_src_len, max_mel_len, embeds = inputs

        embedded_inputs = self.embedding(text).transpose(1, 2)  # [B, 512, L]

        encoder_outputs = self.encoder(
            embedded_inputs, src_len, embeds)  # [B, L, 1280]

        # duration_outputs = self.duration_predictor(
        #     encoder_outputs)  # [B, L, 1]

        range_inputs = torch.cat(
            (encoder_outputs, durations.unsqueeze(-1)), dim=2)  # [B, L, 1281]

        range_outputs = self.range_predictor(range_inputs)  # [B, L, 1]

        # [B, T, 1] [B, T, 1280]
        alignments, sampler_outputs = self.gaussian_upsampling(
            encoder_outputs, durations, range_outputs, self.device)

        pos_enc = self.pos_enc(alignments)  # [B, L, 32]

        memory = torch.cat((sampler_outputs, pos_enc), dim=2)  # [B, T, 1312]
        mel_outputs = self.decoder(
            memory, sampler_outputs, mel_target)  # [B, Mel, T]

        mel_outputs_postnet = self.postnet(mel_outputs)  # [B, Mel, T]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        # print(mel_outputs_postnet.size())
        # return self.parse_output(
        #     [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
        #     output_lengths)

    def inference(self, inputs, embeds):

        embedded_inputs = self.embedding(inputs).transpose(1, 2)  # [B, 512, L]
        encoder_outputs = self.encoder.inference(
            embedded_inputs, embeds)  # [B, L, 1280]
        duration_outputs = self.duration_predictor(
            encoder_outputs)  # [B, L, 1]
        range_inputs = torch.cat(
            (encoder_outputs, duration_outputs), dim=2)  # [B, L, 1281]
        range_outputs = self.range_predictor(range_inputs)  # [B, L, 1]
        # [B, T, 1] [B, T, 1280]
        alignments, sampler_outputs = self.gaussian_upsampling(
            encoder_outputs, duration_outputs.squeeze(-1), range_outputs, self.device)
        pos_enc = self.pos_enc(alignments)  # [B, T, 32]
        memory = torch.cat((sampler_outputs, pos_enc), dim=2)  # [B, T, 1312]
        mel_outputs = self.decoder.inference(
            memory, sampler_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)
