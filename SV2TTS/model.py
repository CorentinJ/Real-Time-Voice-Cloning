from datasets.speaker_batch import SpeakerBatch
from datasets.data_loader import SpeakerVerificationDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import nn
import numpy as np
import torch
from params import *
from config import device

hidden_size = 64
embedding_size = 64
num_layers = 3
learning_rate = 0.0001

speakers_per_batch = 5
utterances_per_speaker = 6

class SpeakerEncoder(nn.Module):
    def __init__(self, train: bool):
        super().__init__()
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=embedding_size)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameters)
        self.similarity_linear = nn.Linear(in_features=1,
                                           out_features=1)
        self.similarity_linear.weight.data = torch.Tensor([[10.]])
        self.similarity_linear.bias.data = torch.Tensor([-5.])
        
        # if train:
        #     self.similarity_matrix =  
        
    
    def forward(self, x, h):
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (h, c) = self.lstm(x, h)
        
        # We take only the hidden state of the last layer
        embed_raw = self.relu(self.linear(h[-1]))
        
        # L2-normalize it
        embed_norms = torch.sqrt(torch.sum(embed_raw ** 2, dim=1, keepdim=True))
        embed = embed_raw / embed_norms
        
        return embed
    
    def similarity(self, embed_a, embed_b):
        assert embed_a.shape == embed_b.shape == (embedding_size,)
        
        # Cosine similarity
        sim_raw = torch.dot(embed_a, embed_b) / (torch.sum(embed_a ** 2) * torch.sum(embed_b ** 2))
        
        # Scale linearly
        sim = self.similarity_linear(sim_raw.view(1))
        
        return sim

    def loss(self, preds):
        ## See section 2.1 of GE2E
        
        preds = preds.view((speakers_per_batch, utterances_per_speaker, embedding_size))
        
        # Inclusive centroids (1 per speaker)
        centroids_incl = torch.mean(preds, dim=1)
        
        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(preds, dim=1, keepdim=True) - preds)
        centroids_excl /= (utterances_per_speaker - 1)
                         
        # Similarity matrix S_(i,j,k)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker, speakers_per_batch)
        for j in range(speakers_per_batch):
            for i in range(utterances_per_speaker):
                for k in range(speakers_per_batch):
                    centroid = centroids_excl[j, i] if j == k else centroids_incl[k]
                    sim_matrix[j, i, k] = self.similarity(preds[j, i], centroid)
        
        # Loss
        loss_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker)
        for j in range(speakers_per_batch):
            for i in range(utterances_per_speaker):
                loss_matrix[j, i] = torch.log(torch.sum(torch.exp(sim_matrix[j, i])))
                loss_matrix[j, i] -= sim_matrix[j, i, j]
        
        return torch.sum(loss_matrix)        


if __name__ == '__main__':
    # from audio import plot_mel_filterbank
    
    dataset = SpeakerVerificationDataset(
        datasets=['train-other-500'],
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
    )
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=1, 
        collate_fn=SpeakerVerificationDataset.collate
    )
    
    model = SpeakerEncoder(train=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for i, speaker_batch in enumerate(loader):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        states = (torch.zeros(num_layers, len(inputs), hidden_size).to(device),
                  torch.zeros(num_layers, len(inputs), hidden_size).to(device))
        outputs = model(inputs, states)
        loss = model.loss(outputs)

        # Backward pass
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if i % 10 == 0:
            print('Step %d: ' % i)
            print('\tAverage loss: %.4f' % np.mean(losses))
            losses = []

            # with torch.no_grad():
            #     test_data = dataset.test_data()
            #     for speaker, partial_utterances in test_data.items():
            #         input = np.array([p[1] for p in partial_utterances])
            #         input = torch.from_numpy(input).to(device)
            #         l = len(partial_utterances)
            #         states = (torch.zeros(1, l, 64).to(device),
            #                   torch.zeros(1, l, 64).to(device))
            #         target = np.array([speaker_dict[speaker]] * l)
            #         target = torch.from_numpy(target).long().to(device)
            # 
            #         outputs = model(input, states)
            # 
            #         pred = torch.argmax(outputs, dim=1)
            #         accuracy = torch.mean((pred == target).float())
            #         accuracies.append(accuracy.item())
            # 
            #         loss = criterion(outputs, target)
            #         losses.append(loss.item())
            # 
            # print('\tVal loss: %.4f' % np.mean(losses))
            # print('\tVal accuracy: %.4f' % np.mean(accuracies))
            # losses = []
            # accuracies = []










