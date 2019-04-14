from encoder.ui.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from pathlib import Path
import torch


def train(run_id: str, clean_data_root: Path, models_dir: Path, vis_every: int, save_every: int,
          backup_every: int, force_restart: bool):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )
    
    # Create the model and the optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerEncoder(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + '.pt')
    backup_dir = models_dir.joinpath(run_id + '_backups')

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print('Found existing model \"%s\", loading it and resuming training.' % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            optimizer.param_groups[0]['lr'] = learning_rate_init
        else:
            print('No model \"%s\" found, starting training from scratch.' % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    vis = Visualizations(run_id, device_name=device_name)
    vis.log_dataset(dataset)
    
    # Training loop
    for step, speaker_batch in enumerate(loader, init_step):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        embeds = model(inputs).cpu()
        loss, eer = model.loss(embeds.view((speakers_per_batch, utterances_per_speaker, -1)))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
        
        # Update visualizations
        learning_rate = optimizer.param_groups[0]['lr']
        vis.update(loss.item(), eer, learning_rate, step)
        
        # Draw projections and save them to the backup folder
        if vis_every != 0 and step % vis_every == 0:
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath('%s_umap_%06d.png' % (run_id, step))
            embeds_numpy = embeds.detach().numpy()
            vis.draw_projections(embeds_numpy, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            torch.save({
                'step': step + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                'step': step + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, backup_fpath)
            