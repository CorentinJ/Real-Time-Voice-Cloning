from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
from data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from ui.visualizations import Visualizations
from config import device
from model import SpeakerEncoder
import numpy as np
import torch


## Training parameters
learning_rate_init = 1e-4
# exponential_decay_beta = 0.9998
speakers_per_batch = 5
utterances_per_speaker = 6

implementation_doc = {
    'Lr decay': None,
    'Gradient ops': True,
    'Projection layer': False,
}

if __name__ == '__main__':
    # Create a data loader
    dataset = SpeakerVerificationDataset(
        datasets=['train-other-500'],
    )
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=1,
    )
    
    # Create the model and the optimizer
    model = SpeakerEncoder(speakers_per_batch, utterances_per_speaker)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exponential_decay_beta)
    
    # Initialize the visualization environment
    vis = Visualizations()
    vis.log_dataset(dataset)
    vis.log_implementation(implementation_doc)
    
    # Training loop
    loss_values = []
    error_rates = []
    for step, speaker_batch in enumerate(loader, 1):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        embeds = model(inputs).cpu()
        loss, eer = model.loss(embeds)
        loss_values.append(loss.item())
        error_rates.append(eer)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
        # scheduler.step()
        
        # Visualization data
        if step % 10 == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            vis.update(np.mean(loss_values), np.mean(error_rates), learning_rate, step)
            loss_values.clear()
            error_rates.clear()
        if step % 100 == 0:
            vis.draw_projections(embeds.detach().numpy(), utterances_per_speaker, step)
