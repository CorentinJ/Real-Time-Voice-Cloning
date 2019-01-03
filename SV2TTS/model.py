from torch.nn.utils import clip_grad_norm_
from config import device
from params import *
from torch import nn
import numpy as np
import torch

class SpeakerEncoder(nn.Module):
    def __init__(self, speakers_per_batch, utterances_per_speaker):
        super().__init__()
        
        # Dimensions of the inputs
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker 
        
        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss and accuracy computation
        self.ground_truth = torch.from_numpy(
            np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        ).long()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (batch_size, embedding_size), 
        where batch_size == speakers_per_batch * utterances_per_speaker and where each speaker's 
        utterances are listed sequentially
        :return: the loss and the accuracy for this batch of embeddings. Note that this accuracy 
        does properly reflect the SV-EER of the model as the centroid of each speaker is computed
        from the predicted embeddings themselves, which inflates the accuracy.
        """
        # Computation is significantly faster on the CPU
        if embeds.device != torch.device('cpu'):
            embeds = embeds.to(torch.device('cpu'))
            
        embeds = embeds.view((
            self.speakers_per_batch, 
            self.utterances_per_speaker, 
            model_embedding_size
        ))
        
        # Inclusive centroids (1 per speaker)
        centroids_incl = torch.mean(embeds, dim=1)
        centroid_incl_norms = torch.norm(centroids_incl, dim=1)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (self.utterances_per_speaker - 1)
        centroid_excl_norms = torch.norm(centroids_excl, dim=2)
                         
        # Similarity matrix
        sim_matrix = torch.zeros(
            self.speakers_per_batch * self.utterances_per_speaker, 
            self.speakers_per_batch
        )
        for j in range(self.speakers_per_batch):
            for i in range(self.utterances_per_speaker):
                ji = j * self.utterances_per_speaker + i
                for k in range(self.speakers_per_batch):
                    centroid = centroids_excl[j, i] if j == k else centroids_incl[k]
                    centroid_norm = centroid_excl_norms[j, i] if j == k else centroid_incl_norms[k]
                    # Note: the sum of squares of the embeddings is always 1 due to the 
                    # L2-normalization, we can thus ignore it. 
                    sim_matrix[ji, k] = torch.dot(embeds[j, i], centroid) / centroid_norm
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias

        # Loss
        loss = self.loss_fn(sim_matrix, self.ground_truth)
        
        # Accuracy (not backpropagated)
        with torch.no_grad():
            preds = torch.argmax(sim_matrix, dim=1)
            accuracy = torch.mean((preds == self.ground_truth).float())
        
        return loss, accuracy