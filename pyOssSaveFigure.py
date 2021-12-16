# Import Systems 
import struct
import io
import os
import sys
import math
import platform

# Import Audio
import pyaudio
import librosa
import soundfile

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssLearn as learn

plt.rc('font', family='Malgun Gothic')			# 한글폰트 사용

save_dir = 'imagefiles'

imp_dir = 'ju_impulse2'
# imp_dir = 'ju_impulse3'
# imp_dir = 'ju_impulse_test'

g_current_impulse_file = ''

fs = 48000

for impulseFilename in os.listdir(imp_dir):
	if impulseFilename.endswith(".wav"):
		impulse_file_directory = os.path.join(imp_dir, impulseFilename)

		print("\n\n\n")
		print("impulse_file_directory = ", impulse_file_directory)

		g_current_impulse_file = impulseFilename       #global

		imp_name = impulseFilename[:-4]

		imp_fname = pyOssWavfile.str_fname(imp_dir, imp_name) # 전체경로, 파일명 조합
		fmt_imp, data_imp, st_fmt_imp, t_imp = pyOssWavfile.readf32(imp_fname, fs)
		dbg.dWavInfo(st_fmt_imp)
		print(f" - Time(sec) =", t_imp)

		# t_estimate = None
		t_estimate = room.estimate_rt(data_imp, fs)
		# t_estimate = 0

		decay_imp = room.decayCurve(data_imp, estimate=t_estimate, fs=fs)
		C_a_param_imp = room.calcAcousticParam(data_imp, decay_imp, fs, label_text=imp_name)
		dbg.dPrintAParam(C_a_param_imp)

		dbg.dSavePlotAudio(	fs, \
							data_imp, \
							title_txt=imp_name, \
							label_txt='RT60='+str(C_a_param_imp.RT60), \
							xl_txt='Time(sec)', \
							yl_txt='Amplitude', \
							newWindow=True, \
							directory='./'+imp_dir+'/'+save_dir )
							
		dbg.dSavePlotDecay(	fs, \
							decay_imp, \
							title_txt=imp_name + '_Decay' , \
							label_txt='RT60='+str(C_a_param_imp.RT60), \
							xl_txt='Time(sec)', \
							yl_txt='Amplitude(dB)', \
							newWindow=True, \
							directory='./'+imp_dir+'/'+save_dir )
