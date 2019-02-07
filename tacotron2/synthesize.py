import argparse
import os
from warnings import warn
from time import sleep

import tensorflow as tf

from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from wavenet_vocoder.synthesize import wavenet_synthesize


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    run_name = args.name or args.tacotron_name or args.model
    taco_checkpoint = os.path.join('logs-' + run_name, 'taco_' + args.checkpoint)
    
    run_name = args.name or args.wavenet_name or args.model
    wave_checkpoint = os.path.join('logs-' + run_name, 'wave_' + args.checkpoint)
    return taco_checkpoint, wave_checkpoint, modified_hp


def get_sentences(args):
    if args.text_list != '':
        with open(args.text_list, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = hparams.sentences
    return sentences


def synthesize(args, hparams, taco_checkpoint, wave_checkpoint, sentences):
    log('Running End-to-End TTS Evaluation. Model: {}'.format(args.name or args.model))
    log('Synthesizing mel-spectrograms from text..')
    wavenet_in_dir = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
    # Delete Tacotron model from graph
    tf.reset_default_graph()
    # Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is 
    # synthesizing
    sleep(0.5)
    log('Synthesizing audio from mel-spectrograms.. (This may take a while)')
    wavenet_synthesize(args, hparams, wave_checkpoint)
    log('Tacotron-2 TTS synthesis complete!')


def main():
    accepted_modes = ['eval', 'synthesis', 'live']
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value '
							 'pairs')
    parser.add_argument('--name',
                        help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--tacotron_name',
                        help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--wavenet_name',
                        help='Name of logging directory of WaveNet. If trained separately')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--input_dir', default='training_data/',
                        help='folder to contain inputs sentences/targets')
    parser.add_argument('--mels_dir', default='tacotron_output/eval/',
                        help='folder to contain mels to synthesize audio from using the Wavenet')
    parser.add_argument('--output_dir', default='output/',
                        help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='eval',
                        help='mode of run: can be one of {}'.format(accepted_modes))
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered '
							 'in synthesis mode')
    parser.add_argument('--text_list', default='',
                        help='Text file contains list of texts to be synthesized. Valid if '
							 'mode=eval')
    parser.add_argument('--speaker_id', default=None,
                        help='Defines the speakers ids to use when running standalone Wavenet on '
							 'a folder of mels. this variable must be a comma-separated list of '
							 'ids')
    args = parser.parse_args()
    
    accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']
    
    if args.model not in accepted_models:
        raise ValueError(
            'please enter a valid model to synthesize with: {}'.format(accepted_models))
    
    if args.mode not in accepted_modes:
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))
    
    if args.mode == 'live' and args.model == 'Wavenet':
        raise RuntimeError(
            'Wavenet vocoder cannot be tested live due to its slow generation. Live only works '
			'with Tacotron!')
    
    if args.GTA not in ('True', 'False'):
        raise ValueError('GTA option must be either True or False')
    
    if args.model == 'Tacotron-2':
        if args.mode == 'live':
            warn('Requested a live evaluation with Tacotron-2, Wavenet will not be used!')
        if args.mode == 'synthesis':
            raise ValueError(
                'I don\'t recommend running WaveNet on entire dataset.. The world might end '
				'before the synthesis :) (only eval allowed)')
    
    taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
    sentences = get_sentences(args)
    
    if args.model == 'Tacotron':
        _ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
    elif args.model == 'WaveNet':
        wavenet_synthesize(args, hparams, wave_checkpoint)
    elif args.model == 'Tacotron-2':
        synthesize(args, hparams, taco_checkpoint, wave_checkpoint, sentences)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
    main()
