import torch
from utils.paths import Paths
from models.tacotron import Tacotron


def get_checkpoint_paths(checkpoint_type: str, paths: Paths):
    """
    Returns the correct checkpointing paths
    depending on whether model is Vocoder or TTS

    Args:
        checkpoint_type: Either 'voc' or 'tts'
        paths: Paths object
    """
    if checkpoint_type is 'tts':
        weights_path = paths.tts_latest_weights
        optim_path = paths.tts_latest_optim
        checkpoint_path = paths.tts_checkpoints
    elif checkpoint_type is 'voc':
        weights_path = paths.voc_latest_weights
        optim_path = paths.voc_latest_optim
        checkpoint_path = paths.voc_checkpoints
    else:
        raise NotImplementedError

    return weights_path, optim_path, checkpoint_path


def save_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, *,
        name=None, is_silent=False):
    """Saves the training session to disk.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will name to a checkpoint with the given name. Note
            that regardless of whether this is provided or not, this function
            will always update the files specified in `paths` that give the
            location of the latest weights and optimizer state. Saving
            a named checkpoint happens in addition to this update.
    """
    def helper(path_dict, is_named):
        s = 'named' if is_named else 'latest'
        num_exist = sum(p.exists() for p in path_dict.values())

        if num_exist not in (0,2):
            # Checkpoint broken
            raise FileNotFoundError(
                f'We expected either both or no files in the {s} checkpoint to '
                'exist, but instead we got exactly one!')

        if num_exist == 0:
            if not is_silent: print(f'Creating {s} checkpoint...')
            for p in path_dict.values():
                p.parent.mkdir(parents=True, exist_ok=True)
        else:
            if not is_silent: print(f'Saving to existing {s} checkpoint...')

        if not is_silent: print(f'Saving {s} weights: {path_dict["w"]}')
        model.save(path_dict['w'])
        if not is_silent: print(f'Saving {s} optimizer state: {path_dict["o"]}')
        torch.save(optimizer.state_dict(), path_dict['o'])

    weights_path, optim_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    latest_paths = {'w': weights_path, 'o': optim_path}
    helper(latest_paths, False)

    if name:
        named_paths = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
        }
        helper(named_paths, True)


def restore_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, *,
        name=None, create_if_missing=False):
    """Restores from a training session saved to disk.

    NOTE: The optimizer's state is placed on the same device as it's model
    parameters. Therefore, be sure you have done `model.to(device)` before
    calling this method.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will restore from a checkpoint with the given name.
            Otherwise, will restore from the latest weights and optimizer state
            as specified in `paths`.
        create_if_missing:  If `True`, will create the checkpoint if it doesn't
            yet exist, as well as update the files specified in `paths` that
            give the location of the current latest weights and optimizer state.
            If `False` and the checkpoint doesn't exist, will raise a
            `FileNotFoundError`.
    """

    weights_path, optim_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    if name:
        path_dict = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
        }
        s = 'named'
    else:
        path_dict = {
            'w': weights_path,
            'o': optim_path
        }
        s = 'latest'

    num_exist = sum(p.exists() for p in path_dict.values())
    if num_exist == 2:
        # Checkpoint exists
        print(f'Restoring from {s} checkpoint...')
        print(f'Loading {s} weights: {path_dict["w"]}')
        model.load(path_dict['w'])
        print(f'Loading {s} optimizer state: {path_dict["o"]}')
        optimizer.load_state_dict(torch.load(path_dict['o']))
    elif create_if_missing:
        save_checkpoint(checkpoint_type, paths, model, optimizer, name=name, is_silent=False)
    else:
        raise FileNotFoundError(f'The {s} checkpoint could not be found!')