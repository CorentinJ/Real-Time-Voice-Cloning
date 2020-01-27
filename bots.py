from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
import slack, asyncio
import string, random

import tweepy
from config import create_api

import time
import boto3
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

api = create_api()
since_id = 1 

def gen_random_str(n):
    """
    Helper function to generate a random string to use
    as a file name of the audio generated.
    """
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))
    return res

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name, ExtraArgs={'ACL':'public-read'})
    except ClientError as e:
        logging.error(e)
        return False
    return True


if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd
        
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)
    
    
   
    
    try:
        # Get the reference audio filepath
        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                  "wav, m4a, flac, ...):\n"
        # in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
        in_fpath = Path("/home/ubuntu/SFry.flac")
        
        
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is 
        # important: there is preprocessing that must be applied.
        
        # The following two methods are equivalent:
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded Stephen Fry reference file succesfully")
        
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")
        
        
        ## Generating the spectrogram
        def gen_sound(text): #text = input("Write a sentence (+-20 words) to be synthesized:\n")
            print(text, "input text is")
        
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            text = text[3:]
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")
            
            
            ## Generating the waveform
            print("Synthesizing the waveform:")
            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)
            
            
            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            
            # Save it on the disk
            fpath = gen_random_str(7) + ".wav"
            print(generated_wav.dtype)
            librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
                                     synthesizer.sample_rate)
            print("\nSaved output as %s\n\n" % fpath)
            upload_file(fpath, "wandbanil")
            # return "https://wandbanil.s3.amazonaws.com/" + fpath
            return fpath


        ###########################################
        ####### SLACK DM BOT ######################
        ###########################################
        @slack.RTMClient.run_on(event='message')
        async def say_hello(**payload):
            data = payload['data']
            web_client = payload['web_client']
            rtm_client = payload['rtm_client']
            if 'Say' in data.get('text', []) and 'user' in data:
                url = gen_sound(data.get('text', []))
                channel_id = data['channel']
                thread_ts = data['ts']
                user = data['user']
               #  await web_client.chat_postMessage(
               #      channel=channel_id,
               #      text=f"Hi <@{user}>! " + url,
               #      thread_ts=thread_ts
               #  )
                await web_client.files_upload(
                    channels=channel_id,
                    file=url,
                    title="Stephen Fry Says",
                    filetype='wav',
                    thread_ts=thread_ts
                )
            elif 'user' in data and 'display_as_bot' not in data:
                channel_id = data['channel']
                thread_ts = data['ts']
                user = data['user']
                await web_client.chat_postMessage(
                    channel=channel_id,
                    text=f"Hi <@{user}>! If you want to use this bot, use the command Say and type something that you want to hear in Stephen Fry's voice",
                    thread_ts=thread_ts
                )
        
        my_slack_token = "your slack user bot token"
        loop = asyncio.get_event_loop()
        rtm_client = slack.RTMClient(token=my_slack_token, run_async=True, loop=loop)
        loop.run_until_complete(rtm_client.start())

        print("Loop Started")
        
         
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
    
