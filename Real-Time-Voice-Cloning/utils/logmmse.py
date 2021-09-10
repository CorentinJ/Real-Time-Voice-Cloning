# The MIT License (MIT)
# 
# Copyright (c) 2015 braindead
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# This code was extracted from the logmmse package (https://pypi.org/project/logmmse/) and I
# simply modified the interface to meet my needs.


import numpy as np
import math
from scipy.special import expn
from collections import namedtuple

NoiseProfile = namedtuple("NoiseProfile", "sampling_rate window_size len1 len2 win n_fft noise_mu2")


def profile_noise(noise, sampling_rate, window_size=0):
    """
    Creates a profile of the noise in a given waveform.
    
    :param noise: a waveform containing noise ONLY, as a numpy array of floats or ints. 
    :param sampling_rate: the sampling rate of the audio
    :param window_size: the size of the window the logmmse algorithm operates on. A default value 
    will be picked if left as 0.
    :return: a NoiseProfile object
    """
    noise, dtype = to_float(noise)
    noise += np.finfo(np.float64).eps

    if window_size == 0:
        window_size = int(math.floor(0.02 * sampling_rate))

    if window_size % 2 == 1:
        window_size = window_size + 1
    
    perc = 50
    len1 = int(math.floor(window_size * perc / 100))
    len2 = int(window_size - len1)

    win = np.hanning(window_size)
    win = win * len2 / np.sum(win)
    n_fft = 2 * window_size

    noise_mean = np.zeros(n_fft)
    n_frames = len(noise) // window_size
    for j in range(0, window_size * n_frames, window_size):
        noise_mean += np.absolute(np.fft.fft(win * noise[j:j + window_size], n_fft, axis=0))
    noise_mu2 = (noise_mean / n_frames) ** 2
    
    return NoiseProfile(sampling_rate, window_size, len1, len2, win, n_fft, noise_mu2)


def denoise(wav, noise_profile: NoiseProfile, eta=0.15):
    """
    Cleans the noise from a speech waveform given a noise profile. The waveform must have the 
    same sampling rate as the one used to create the noise profile. 
    
    :param wav: a speech waveform as a numpy array of floats or ints.
    :param noise_profile: a NoiseProfile object that was created from a similar (or a segment of 
    the same) waveform.
    :param eta: voice threshold for noise update. While the voice activation detection value is 
    below this threshold, the noise profile will be continuously updated throughout the audio. 
    Set to 0 to disable updating the noise profile.
    :return: the clean wav as a numpy array of floats or ints of the same length.
    """
    wav, dtype = to_float(wav)
    wav += np.finfo(np.float64).eps
    p = noise_profile
    
    nframes = int(math.floor(len(wav) / p.len2) - math.floor(p.window_size / p.len2))
    x_final = np.zeros(nframes * p.len2)

    aa = 0.98
    mu = 0.98
    ksi_min = 10 ** (-25 / 10)
    
    x_old = np.zeros(p.len1)
    xk_prev = np.zeros(p.len1)
    noise_mu2 = p.noise_mu2
    for k in range(0, nframes * p.len2, p.len2):
        insign = p.win * wav[k:k + p.window_size]

        spec = np.fft.fft(insign, p.n_fft, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2

        gammak = np.minimum(sig2 / noise_mu2, 40)

        if xk_prev.all() == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(ksi_min, ksi)

        log_sigma_k = gammak * ksi/(1 + ksi) - np.log(1 + ksi)
        vad_decision = np.sum(log_sigma_k) / p.window_size
        if vad_decision < eta:
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2

        a = ksi / (1 + ksi)
        vk = a * gammak
        ei_vk = 0.5 * expn(1, np.maximum(vk, 1e-8))
        hw = a * np.exp(ei_vk)
        sig = sig * hw
        xk_prev = sig ** 2
        xi_w = np.fft.ifft(hw * spec, p.n_fft, axis=0)
        xi_w = np.real(xi_w)

        x_final[k:k + p.len2] = x_old + xi_w[0:p.len1]
        x_old = xi_w[p.len1:p.window_size]

    output = from_float(x_final, dtype)
    output = np.pad(output, (0, len(wav) - len(output)), mode="constant")
    return output


## Alternative VAD algorithm to webrctvad. It has the advantage of not requiring to install that 
## darn package and it also works for any sampling rate. Maybe I'll eventually use it instead of 
## webrctvad
# def vad(wav, sampling_rate, eta=0.15, window_size=0):
#     """
#     TODO: fix doc
#     Creates a profile of the noise in a given waveform.
# 
#     :param wav: a waveform containing noise ONLY, as a numpy array of floats or ints. 
#     :param sampling_rate: the sampling rate of the audio
#     :param window_size: the size of the window the logmmse algorithm operates on. A default value 
#     will be picked if left as 0.
#     :param eta: voice threshold for noise update. While the voice activation detection value is 
#     below this threshold, the noise profile will be continuously updated throughout the audio. 
#     Set to 0 to disable updating the noise profile.
#     """
#     wav, dtype = to_float(wav)
#     wav += np.finfo(np.float64).eps
#     
#     if window_size == 0:
#         window_size = int(math.floor(0.02 * sampling_rate))
#     
#     if window_size % 2 == 1:
#         window_size = window_size + 1
#     
#     perc = 50
#     len1 = int(math.floor(window_size * perc / 100))
#     len2 = int(window_size - len1)
#     
#     win = np.hanning(window_size)
#     win = win * len2 / np.sum(win)
#     n_fft = 2 * window_size
#     
#     wav_mean = np.zeros(n_fft)
#     n_frames = len(wav) // window_size
#     for j in range(0, window_size * n_frames, window_size):
#         wav_mean += np.absolute(np.fft.fft(win * wav[j:j + window_size], n_fft, axis=0))
#     noise_mu2 = (wav_mean / n_frames) ** 2
#     
#     wav, dtype = to_float(wav)
#     wav += np.finfo(np.float64).eps
#     
#     nframes = int(math.floor(len(wav) / len2) - math.floor(window_size / len2))
#     vad = np.zeros(nframes * len2, dtype=np.bool)
# 
#     aa = 0.98
#     mu = 0.98
#     ksi_min = 10 ** (-25 / 10)
#     
#     xk_prev = np.zeros(len1)
#     noise_mu2 = noise_mu2
#     for k in range(0, nframes * len2, len2):
#         insign = win * wav[k:k + window_size]
#         
#         spec = np.fft.fft(insign, n_fft, axis=0)
#         sig = np.absolute(spec)
#         sig2 = sig ** 2
#         
#         gammak = np.minimum(sig2 / noise_mu2, 40)
#         
#         if xk_prev.all() == 0:
#             ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
#         else:
#             ksi = aa * xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
#             ksi = np.maximum(ksi_min, ksi)
#         
#         log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
#         vad_decision = np.sum(log_sigma_k) / window_size
#         if vad_decision < eta:
#             noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2
#         print(vad_decision)
#         
#         a = ksi / (1 + ksi)
#         vk = a * gammak
#         ei_vk = 0.5 * expn(1, np.maximum(vk, 1e-8))
#         hw = a * np.exp(ei_vk)
#         sig = sig * hw
#         xk_prev = sig ** 2
#         
#         vad[k:k + len2] = vad_decision >= eta
#         
#     vad = np.pad(vad, (0, len(wav) - len(vad)), mode="constant")
#     return vad


def to_float(_input):
    if _input.dtype == np.float64:
        return _input, _input.dtype
    elif _input.dtype == np.float32:
        return _input.astype(np.float64), _input.dtype
    elif _input.dtype == np.uint8:
        return (_input - 128) / 128., _input.dtype
    elif _input.dtype == np.int16:
        return _input / 32768., _input.dtype
    elif _input.dtype == np.int32:
        return _input / 2147483648., _input.dtype
    raise ValueError('Unsupported wave file format')


def from_float(_input, dtype):
    if dtype == np.float64:
        return _input, np.float64
    elif dtype == np.float32:
        return _input.astype(np.float32)
    elif dtype == np.uint8:
        return ((_input * 128) + 128).astype(np.uint8)
    elif dtype == np.int16:
        return (_input * 32768).astype(np.int16)
    elif dtype == np.int32:
        print(_input)
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format')
