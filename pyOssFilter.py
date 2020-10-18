'''
# 라이브러리 정의
 - 시스템 라이브러리
 - 오디오 관련 라이브러리
 - 연산 및 Plot 관련 라이브러리 
 - 샤용자 라이브러리
'''

# Import Systems 
import struct
import io
import os
import sys
import time
import math

# Import Audio
import wave
import pyaudio
import librosa

import numpy
import scipy
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.animation
import soundfile

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg




def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    if highcut >= nyq:
        #high = (nyq - 1) / nyq
        high = 0.99
    else:
        high = highcut / nyq

    print(high)
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filtered = signal.lfilter(b, a, data)
    return filtered

def bp_cf_to_band_range(fc, octave=0):
    if octave == 3:
        f1 = math.trunc(fc / (math.sqrt(2**(1/3))))
        f2 = math.trunc(fc * (math.sqrt(2**(1/3))))
    else:
        f1 = math.trunc(0.707 * fc)
        f2 = math.trunc(1.414 * fc)
    #print(f1, f2)

    return f1, f2

'''
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta

# Several flavors of bandpass FIR filters.

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps

def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5*fs]
    taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
    return taps


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 63.0
    lowcut = 0.7
    highcut = 4.0

    ntaps = 128
    taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
    taps_kaiser16 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.6)
    taps_kaiser10 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)
    remez_width = 1.0
    taps_remez = bandpass_remez(ntaps, lowcut, highcut, fs=fs,
                                width=remez_width)

    # Plot the frequency responses of the filters.
    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((lowcut, 0), highcut - lowcut, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = freqz(taps_hamming, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Hamming window")

    w, h = freqz(taps_kaiser16, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.6")

    w, h = freqz(taps_kaiser10, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.0")

    w, h = freqz(taps_remez, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h),
             label="Remez algorithm, width=%.1f" % remez_width)

    plt.xlim(0, 8.0)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of several FIR filters, %d taps' % ntaps)

    plt.show()
    '''