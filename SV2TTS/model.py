from datasets.data_loader import SpeakerVerificationDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import nn
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 64
num_layers = 1
learning_rate = 0.001

# RNN based language model
class SpeakerEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.lstm = nn.LSTM(40, 64, 1, batch_first=True)
        self.linear = nn.Linear(64, 10)
    
    def forward(self, x, h):
        out, (h, c) = self.lstm(x, h)
        h = h.squeeze()
        out = self.linear(h)
        return out

if __name__ == '__main__':
    # from audio import plot_mel_filterbank
    
    dataset = SpeakerVerificationDataset(['train-other-500'], 3, 4)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, 
                        collate_fn=SpeakerVerificationDataset.collate)
    speaker_dict = {speaker.name: i for i, speaker in enumerate(dataset.speakers)}
    
    model = SpeakerEncoder().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    accuracies = []
    for i, batches in enumerate(loader):
        speaker_batch = batches[0]
        # plot_mel_filterbank(speaker_batch[0].numpy(), 16000)
        
        input = torch.from_numpy(speaker_batch.data).to(device)
        target = [speaker_dict[s.name] for s in speaker_batch.speakers]
        target = torch.from_numpy(np.repeat(target, 4)).long().to(device)

        # Set initial hidden and cell states
        states = (torch.zeros(1, 12, 64).to(device),
                  torch.zeros(1, 12, 64).to(device))

        outputs = model(input, states)

        pred = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((pred==target).float())
        accuracies.append(accuracy.item())
        
        loss = criterion(outputs, target)
        losses.append(loss.item())
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if i % 100 == 0:
            print('Step %d: ' % i)
            print('\tAverage loss: %.4f' % np.mean(losses))
            print('\tAverage accuracy: %.4f' % np.mean(accuracies))
            losses = []
            accuracies = []

            with torch.no_grad():
                test_data = dataset.test_data()
                for speaker, partial_utterances in test_data.items():
                    input = np.array([p[1] for p in partial_utterances])
                    input = torch.from_numpy(input).to(device)
                    l = len(partial_utterances)
                    states = (torch.zeros(1, l, 64).to(device),
                              torch.zeros(1, l, 64).to(device))
                    target = np.array([speaker_dict[speaker]] * l)
                    target = torch.from_numpy(target).long().to(device)

                    outputs = model(input, states)

                    pred = torch.argmax(outputs, dim=1)
                    accuracy = torch.mean((pred == target).float())
                    accuracies.append(accuracy.item())

                    loss = criterion(outputs, target)
                    losses.append(loss.item())
                    
            print('\tVal loss: %.4f' % np.mean(losses))
            print('\tVal accuracy: %.4f' % np.mean(accuracies))
            losses = []
            accuracies = []
















# # Test the model
# with torch.no_grad():
#     with open('sample.txt', 'w') as f:
#         # Set intial hidden ane cell states
#         state = (torch.zeros(num_layers, 1, hidden_size).to(device),
#                  torch.zeros(num_layers, 1, hidden_size).to(device))
#         
#         # Select one word id randomly
#         prob = torch.ones(vocab_size)
#         input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
#         
#         for i in range(num_samples):
#             # Forward propagate RNN 
#             output, state = model(input, state)
#             
#             # Sample a word id
#             prob = output.exp()
#             word_id = torch.multinomial(prob, num_samples=1).item()
#             
#             # Fill input with sampled word id for the next time step
#             input.fill_(word_id)
#             
#             if (i + 1) % 100 == 0:
#                 print(
#                     'Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))
# torch.save(model.state_dict(), 'model.ckpt')
