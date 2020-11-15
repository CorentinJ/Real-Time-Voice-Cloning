import sounddevice as sd
import soundfile as sf
from warnings import filterwarnings, warn
import numpy as np
from time import sleep


class Audio:
    def __init__(self, ui, sample_rate):
        self.ui = ui
        self.sample_rate = sample_rate

    def setup_audio_devices(self):
        input_devices = []
        output_devices = []
        for device in sd.query_devices():
            # Check if valid input
            try:
                sd.check_input_settings(
                    device=device["name"], samplerate=self.sample_rate
                )
                input_devices.append(device["name"])
            except:
                pass

            # Check if valid output
            try:
                sd.check_output_settings(
                    device=device["name"], samplerate=self.sample_rate
                )
                output_devices.append(device["name"])
            except Exception as e:
                # Log a warning only if the device is not an input
                if not device["name"] in input_devices:
                    warn(
                        "Unsupported output device %s for the sample rate: %d \nError: %s"
                        % (device["name"], self.sample_rate, str(e))
                    )

        return input_devices, output_devices

    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.ui.log(
                "Error in audio playback. Try selecting a different audio output device."
            )
            self.ui.log("Your device must be connected before you start the toolbox.")

    def stop(self):
        sd.stop()

    def set_audio_device(self, in_dev, out_dev):
        # If None, sounddevice queries portaudio
        sd.default.device = (in_dev, out_dev)

    def record_one(self, sample_rate, duration):
        # self.log("Recording %d seconds of audio" % duration)
        print("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            print("Could not record anything. Is your recording device enabled?")
            print("Your device must be connected before you start the toolbox.")
            # self.log("Could not record anything. Is your recording device enabled?")
            # self.log("Your device must be connected before you start the toolbox.")
            return None

        for i in np.arange(0, duration, 0.1):
            self.ui.set_loading(i, duration)
            sleep(0.1)
        self.ui.set_loading(duration, duration)
        sd.wait()

        # self.log("Done recording.")
        print("Done")
        return wav.squeeze()

    def save_audio(self, fpath, wav, sample_rate):
        sf.write(fpath, wav, sample_rate)
