from torch.nn.utils import clip_grad_norm_
from torch import nn
from params import *
import torch


class SpeakerEncoder(nn.Module):
    def __init__(self, speakers_per_batch, utterances_per_speaker):
        super().__init__()
        
        # Params
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker 
        
        # Network
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameters)
        self.similarity_linear = nn.Linear(in_features=1,
                                           out_features=1)
        self.similarity_linear.weight.data = torch.Tensor([[10.]])
        self.similarity_linear.bias.data = torch.Tensor([-5.])
        
        # Loss and accuracy computation
        self.true_ground = torch.tensor(
            [[i] * utterances_per_speaker for i in range(speakers_per_batch)]
        )
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')
        
    def do_gradient_ops(self):
        # Gradient scale
        for parameters in self.similarity_linear.parameters():
            parameters.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, x, h=None):
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
        assert embed_a.shape == embed_b.shape == (model_embedding_size,)
        
        # Cosine similarity
        sim_raw = torch.dot(embed_a, embed_b) / (torch.sum(embed_a ** 2) * torch.sum(embed_b ** 2))
        
        # Scale linearly
        sim = self.similarity_linear(sim_raw.view(1))
        
        return sim

    def loss(self, embeds):
        ## See section 2.1 of GE2E
        
        embeds = embeds.view((
            self.speakers_per_batch, 
            self.utterances_per_speaker, 
            model_embedding_size
        ))
        
        # Inclusive centroids (1 per speaker)
        centroids_incl = torch.mean(embeds, dim=1)
        
        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (self.utterances_per_speaker - 1)
                         
        # Similarity matrix
        sim_matrix = torch.zeros(
            self.speakers_per_batch, 
            self.utterances_per_speaker, 
            self.speakers_per_batch
        )
        for j in range(self.speakers_per_batch):
            for i in range(self.utterances_per_speaker):
                for k in range(self.speakers_per_batch):
                    centroid = centroids_excl[j, i] if j == k else centroids_incl[k]
                    sim_matrix[j, i, k] = self.similarity(embeds[j, i], centroid)

        # Loss
        loss_vector = torch.zeros(self.speakers_per_batch)
        for j in range(self.speakers_per_batch):
            loss_vector[j] = self.loss_function(sim_matrix[j], self.true_ground[j])
        loss = torch.sum(loss_vector)
        
        # Accuracy (not backpropagated)
        with torch.no_grad():
            preds = torch.argmax(sim_matrix, dim=2)
            accuracy = torch.mean((preds == self.true_ground).float())
        
        return loss, accuracy