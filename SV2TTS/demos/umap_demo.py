from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
from data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from ui.umap_demo_ui import UMapDemoUI
from config import device, model_dir
from model import SpeakerEncoder
from vlibs import fileio
import torch

run_id = 'first_run_64x10'
utterances_per_speaker = 5

if __name__ == '__main__':
    # Create a data loader
    dataset = SpeakerVerificationDataset(
        datasets=['train-other-500'],
    )
    loader = SpeakerVerificationDataLoader(
        dataset,
        1,
        utterances_per_speaker,
        num_workers=2,
    ) 
    
    # Load the model
    model = SpeakerEncoder()
    model_fpath = fileio.join(model_dir, run_id + '.pt')
    checkpoint = torch.load(model_fpath)
    model.load_state_dict(checkpoint['model_state'])

    def get_embeds(speaker_batch, data=None):
        with torch.no_grad():
            if data is None:
                inputs = torch.from_numpy(speaker_batch.data).to(device)
            else:
                inputs = torch.from_numpy(data).to(device)
            embeds = model(inputs).cpu().detach().numpy()
            return embeds
            # loss, eer = model.loss(embeds.view((n_speakers, utterances_per_speaker, -1)))
        
    UMapDemoUI(loader.__iter__(), get_embeds)
