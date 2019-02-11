from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
from data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from ui.visualizations import Visualizations
from params_model import *
from config import *
from model import SpeakerEncoder
from vlibs import fileio
import torch

# Specify the run ID here. Note: visdom will group together run IDs starting with the same prefix
# followed by an underscore.
run_id = None
run_id = 'debug_eer2'
run_id = 'all'

implementation_doc = {
    'Lr decay': None,
    'Gradient ops': True,
    'Projection layer': False,
    'Run ID': run_id,
    'Device': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'),
}

if __name__ == '__main__':
    # Create a data loader
    dataset = SpeakerVerificationDataset(
        datasets=all_datasets,
    )
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )

    # Create the model and the optimizer
    model = SpeakerEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
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
            optimizer.param_groups[0]['lr'] = learning_rate_init
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
        
        # Save state and draw projections
        if step % 100 == 0:
            proj_fpath = None
            if model_fpath is not None:
                # Overwrite the latest version of the model
                fileio.ensure_dir(model_dir)
                torch.save({
                    'step': step + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, model_fpath)
                
                # Make a backup every 2000 steps
                backup_dir = fileio.ensure_dir(fileio.join(model_dir, run_id + '_backups'))
                backup_fpath = fileio.join(backup_dir, "%s_bak_%06d.pt" % (run_id, step))
                if step % 2000 == 0:
                    torch.save({
                        'step': step + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, backup_fpath)

                proj_fpath = fileio.join(backup_dir, run_id)
            vis.draw_projections(embeds.detach().numpy(), utterances_per_speaker, step, proj_fpath)
            vis.save()