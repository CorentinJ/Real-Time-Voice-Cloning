from datasets.speaker_batch import SpeakerBatch
from datasets.speaker_verification_dataset import SpeakerVerificationDataset
from ui.visualizations import Visualizations
from torch.utils.data import DataLoader
from config import device
from model import SpeakerEncoder
import numpy as np
import torch


## Training parameters
learning_rate = 0.0001
speakers_per_batch = 5
utterances_per_speaker = 6

implementation_doc = {
    'Lr decay': None,
    'Gradients ops': False,
    'Projection layer': False,
}

if __name__ == '__main__':
    # Create a data loader
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
    
    # Create the model and the optimizer
    model = SpeakerEncoder(speakers_per_batch, utterances_per_speaker).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the visualization environment
    vis = Visualizations()
    vis.log_dataset(dataset)
    vis.log_implementation(implementation_doc)
    
    # Training loop
    loss_values = []
    for step, speaker_batch in enumerate(loader):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        outputs = model(inputs)
        loss = model.loss(outputs)
        loss_values.append(loss.item())
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        # model.do_gradient_ops()
        optimizer.step()
        
        # Visualization data
        if step % 10 == 0:
            print('Step %d: ' % step)
            print('\tAverage loss: %.4f' % np.mean(loss_values))

            vis.update(np.mean(loss_values), learning_rate, step)
            
            loss_values.clear()


