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
import pandas as pd

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssLearn as learn

plt.rc('font', family='Malgun Gothic')			# 한글폰트 사용

excel_list = []

save_dir = 'excelfiles'

# imp_dir = 'ju_impulse2'
imp_dir = 'ju_impulse3'
# imp_dir = 'ju_impulse_test'

# STAT_RT_ESTIMATE = 'full time'
STAT_RT_ESTIMATE = 'estimate_rt()'

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

		estimate_type = STAT_RT_ESTIMATE

		if STAT_RT_ESTIMATE == 'full time':
			t_estimate = None
			estimate_time = t_imp
		elif STAT_RT_ESTIMATE == 'estimate_rt()':
			t_estimate = room.estimate_rt(data_imp, fs)
			estimate_time = t_estimate

		decay_imp = room.decayCurve(data_imp, estimate=t_estimate, fs=fs)
		C_a_param_imp = room.calcAcousticParam(data_imp, decay_imp, fs, label_text=imp_name)
		dbg.dPrintAParam(C_a_param_imp)

		if estimate_time >= 0.01:
			estimate_time = round(estimate_time, 2)

		if (C_a_param_imp.RT60 >= 0.01) or (C_a_param_imp.EDT >= 0.01):
			C_a_param_imp.RT60 = round(C_a_param_imp.RT60, 2)
			C_a_param_imp.EDT = round(C_a_param_imp.EDT, 2)

		if abs(C_a_param_imp.D50) >= 0.01:
			C_a_param_imp.D50 = round(C_a_param_imp.D50, 2)

		if (abs(C_a_param_imp.C50) >= 0.01) or (abs(C_a_param_imp.C80) >= 0.01):
			C_a_param_imp.C50 = round(C_a_param_imp.C50, 2)
			C_a_param_imp.C80 = round(C_a_param_imp.C80, 2)


		excel_list.append([imp_name, round(t_imp, 2), estimate_type, estimate_time, 
							C_a_param_imp.RT60, C_a_param_imp.EDT, C_a_param_imp.D50,
							C_a_param_imp.C50, C_a_param_imp.C80])

	# 전체 끝난 후 결과 저장
	df = pd.DataFrame(excel_list, columns=['File Name', 'Time of File', 'Estimate Type', 'Estimate Time', 
											'RT60',	'EDT', 'D50',
											'C50', 'C80'])
	df.to_excel('./'+ imp_dir + '/excel_impulse_info_'+ imp_dir + '_' + estimate_type + '.xlsx')
