from torch import nn
import torch
from synthesizer.hparams import hparams


class NATLoss(nn.Module):
    def __init__(self):
        super(NATLoss, self).__init__()
        # self.mse_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, model_output, targets):
        mel_target, duration_target = targets[0].transpose(
            1, 2),  targets[1].float()
        mel_target.requires_grad = False
        duration_target.requires_grad = False

        mel_out, mel_out_postnet, duration = model_output
        duration = duration.float()

        mel_loss1 = self.mae_loss(mel_out, mel_target)
        mel_loss2 = self.mse_loss(mel_out, mel_target)

        mel_postnet_loss1 = self.mae_loss(mel_out_postnet, mel_target)
        mel_postnet_loss2 = self.mse_loss(mel_out_postnet, mel_target)

        mel_loss = mel_loss1 + mel_loss2
        mel_postnet_loss = mel_postnet_loss1 + mel_postnet_loss2

        duration_loss = self.mse_loss(duration, duration_target)

        return mel_loss, mel_postnet_loss, duration_loss


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
