# Import Library for Audio
import sounddevice as sd
import pyaudio
import wave
# Import Library for math & plot
import numpy as np
import matplotlib.pyplot as plt
# Import 
import os
import struct

#%matplotlib tk

CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

print p.get_host_api_info_by_index(1)


