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
import platform

# Import Audio
import wave
import pyaudio
import librosa

import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile
import rir_generator.c_ext as rir
# import rir_generator.c_ext as rir_c_ext        # rir_Generator
# import rir_generator.rir_generator as rir_gen  # rir_Generator

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter


# example_2.m

c = 340                         # Sound velocity (m/s)
fs = 44100                      # Sample frequency (samples/s)
r = [2, 1.5, 2]                 # Receiver position [x y z] (m)
s = [2, 3.5, 2]                 # Source position [x y z] (m)
L = [5, 4, 6]                     # Room dimensions [x y z] (m)
beta = 0.4;                     # Reflections Coefficients
n = 2048;                       # Number of samples
mtype = 'omnidirectional'       # Type of microphone
order = 2                       # Reflection order
dim = 3                         # Room dimension
orientation = 0                 # Microphone orientation (rad)
hp_filter = 1                   # Enable high-pass filter

h2 = rir.generate_rir_ext(c, fs, r, s, L, beta, n, mtype, order, dim, orientation, hp_filter)

