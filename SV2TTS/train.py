from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
from data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from ui.visualizations import Visualizations
from params_model import *
from config import device, model_dir
from model import SpeakerEncoder
from vlibs import fileio
import numpy as np
import torch

# Specify the run ID here. Note: visdom will group together run IDs starting with the same prefix
# followed by an underscore.
run_id = None
run_id = 'save_test'

implementation_doc = {
    'Lr decay': None,
    'Gradient ops': True,
    'Projection layer': False,
    'Run ID': run_id,
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
    # test_loader = SpeakerVerificationDataLoader(
    #     dataset,
    #     64,
    #     2,
    #     num_workers=1,
    # ) 
    
    # Create the model and the optimizer
    model = SpeakerEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exponential_decay_beta)
    init_step = 1
    
    # Load any existing model
    if run_id is not None:
        model_fpath = fileio.join(model_dir, run_id + '.pt')
        if fileio.exists(model_fpath):
            print('Found existing model \"%s\", loading it and resuming training.' % run_id)
            checkpoint = torch.load(model_fpath)
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            print('No model \"%s\" found, starting training from scratch.' % run_id)
    else:
        model_fpath = None
        print('No run ID specified, the model will not be saved or loaded.')
    
    # Initialize the visualization environment
    vis = Visualizations(run_id)
    vis.log_dataset(dataset)
    vis.log_implementation(implementation_doc)
    
    # Training loop
    loss_values = []
    error_rates = []
    for step, speaker_batch in enumerate(loader, init_step):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        embeds = model(inputs).cpu()
        loss, eer = model.loss(embeds.view((speakers_per_batch, utterances_per_speaker, -1)))
        loss_values.append(loss.item())
        error_rates.append(eer)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
        # scheduler.step()
        
        # Update visualizations
        if step % 10 == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            vis.update(np.mean(loss_values), np.mean(error_rates), learning_rate, step)
            loss_values.clear()
            error_rates.clear()
            
        # Draw projections, save state
        if step % 100 == 0:
            vis.draw_projections(embeds.detach().numpy(), utterances_per_speaker, step)
            
            if model_fpath is None:
                continue
            fileio.ensure_dir(model_dir)
            torch.save({
                'step': step + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, model_fpath)
            
            