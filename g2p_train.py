# os.chdir("synthesizer/models/tacotron/utils/g2p")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from synthesizer.g2p import PersianLexicon, collate_fn
from synthesizer.g2p import Encoder, Decoder
from synthesizer.g2p import DataConfig, ModelConfig, TrainConfig

# data prep
ds = PersianLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path
)
dl = DataLoader(
    ds,
    collate_fn=collate_fn,
    batch_size=TrainConfig.batch_size
)

# models
encoder_model = (
    Encoder(
        ModelConfig.graphemes_size,
        ModelConfig.hidden_size
    )
    .to(TrainConfig.device)
)
decoder_model = (
    Decoder(
        ModelConfig.phonemes_size,
        ModelConfig.hidden_size
    )
    .to(TrainConfig.device)
)

# log
log = SummaryWriter(TrainConfig.log_path)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) + list(decoder_model.parameters()),
    lr=TrainConfig.lr
)

# training loop
counter = 0
for e in range(TrainConfig.epochs):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for g, p in tqdm(dl):
        g = g.to(TrainConfig.device)
        p = p.to(TrainConfig.device)
        # encode
        enc = encoder_model(g)

        # decoder
        T, N = p.size()
        outputs = []
        hidden = (
            torch.ones(
                1,
                N,
                ModelConfig.hidden_size
            )
            .to(TrainConfig.device)
        )
        for t in range(T - 1):
            out, hidden, _ = decoder_model(
                p[t:t+1],
                enc,
                hidden
            )
            outputs.append(out)
        outputs = torch.cat(outputs)

        # flat Time and Batch, calculate loss
        outputs = outputs.view((T-1) * N, -1)
        p = p[1:]  # trim first phoneme
        p = p.view(-1)
        loss = criterion(outputs, p)

        # updata weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar('loss', loss.item(), counter)
        counter += 1

    # save model
    torch.save(
        encoder_model.state_dict(),
        f'models/{DataConfig.language}/encoder_e{e+1:02d}.pth'
    )
    torch.save(
        decoder_model.state_dict(),
        f'models/{DataConfig.language}/decoder_e{e+1:02d}.pth'
    )