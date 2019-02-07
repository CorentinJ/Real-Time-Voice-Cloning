from .wavenet import WaveNet
from warnings import warn
from wavenet_vocoder.util import is_mulaw_quantize

def create_model(name, hparams, init=False):
	if is_mulaw_quantize(hparams.input_type):
		if hparams.out_channels != hparams.quantize_channels:
			raise RuntimeError(
				"out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
	if hparams.upsample_conditional_features and hparams.cin_channels < 0:
		s = "Upsample conv layers were specified while local conditioning disabled. "
		s += "Notice that upsample conv layers will never be used."
		warn(s)

	if name == 'WaveNet':
		return WaveNet(hparams, init)
	else:
		raise Exception('Unknow model: {}'.format(name))
