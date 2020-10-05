
# Import Systems 
import struct
import io
import os
import sys
import time

# Import Audio Libraries
import pyaudio

# import Array & Math  
import numpy as np
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt

# User Libraries
import pyOssWavfile
import pyRoomAcoustic


def convPaSampleFormat(aud_fmt, bitdepth):
	"""
	Covert from AUdio Format & Bit per sample in wav format chunk to pyaudio sample format  

	----
	return
		r: pyaudio format
	"""
	
	if aud_fmt == 3 and bitdepth == 32:
		r = pyaudio.paFloat32
	elif aud_fmt <= 2 and aud_fmt > 0 and bitdepth == 16:
		r = pyaudio.paInt16
	else:
		print("Unknown Format. Set paInt16 as default value")
		r = pyaudio.paInt16

	return r

