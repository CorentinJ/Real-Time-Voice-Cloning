import argparse
import os
from time import sleep
import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train

log = infolog.log


def save_seq(file, sequence, input_path):
    '''Save Tacotron-2 training state to disk. (To skip for future runs)
    '''
    sequence = [str(int(s)) for s in sequence] + [input_path]
    with open(file, 'w') as f:
        f.write('|'.join(sequence))


def read_seq(file):
    '''Load Tacotron-2 training state from disk. (To skip if not first run)
    '''
    if os.path.isfile(file):
        with open(file, 'r') as f:
            sequence = f.read().split('|')
        
        return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
    else:
        return [0, 0, 0], ''


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
    return log_dir, modified_hp


def train(args, log_dir, hparams):
    state_file = os.path.join(log_dir, 'state_log')
    # Get training states
    (taco_state, GTA_state, wave_state), input_path = read_seq(state_file)
    taco_state = False
    if not taco_state:
        log('\n#############################################################\n')
        log('Tacotron Train\n')
        log('###########################################################\n')
        checkpoint = tacotron_train(args, log_dir, hparams)
        tf.reset_default_graph()
        # Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
        sleep(0.5)
        if checkpoint is None:
            raise ('Error occured while training Tacotron, Exiting!')
        taco_state = 1
        save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
    else:
        checkpoint = os.path.join(log_dir, 'taco_pretrained/')
    
    if not GTA_state:
        log('\n#############################################################\n')
        log('Tacotron GTA Synthesis\n')
        log('###########################################################\n')
        input_path = tacotron_synthesize(args, hparams, checkpoint)
        tf.reset_default_graph()
        # Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is 
        # training
        sleep(0.5)
        GTA_state = 1
        save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
    else:
        input_path = os.path.join('tacotron_' + args.output_dir, 'gta', 'map.txt')
    
    if input_path == '' or input_path is None:
        raise RuntimeError('input_path has an unpleasant value -> {}'.format(input_path))

    if GTA_state and taco_state:
        log('TRAINING IS ALREADY COMPLETE!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value '
							 'pairs')
    parser.add_argument('--tacotron_input', default='Synthesizer/train.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--input_dir', default='Synthesizer',
                        help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output',
                        help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='synthesis',
                        help='mode for synthesis of tacotron after training')
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered '
							 'in Tacotron synthesis mode')
    parser.add_argument('--restore', type=bool, default=True,
                        help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=2500,
                        help='Steps between running summary ops')
    parser.add_argument('--embedding_interval', type=int, default=10000,
                        help='Steps between updating embeddings projection visualization')
    parser.add_argument('--checkpoint_interval', type=int, default=2000, # Was 5000
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=500000, # Was 10000
                        help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_steps', type=int, default=1000000, # Was 100000
                        help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--slack_url', default=None,
                        help='slack webhook notification destination link')
    args = parser.parse_args()
    
    log_dir, hparams = prepare_run(args)
    
    tacotron_train(args, log_dir, hparams)

if __name__ == '__main__':
    main()
