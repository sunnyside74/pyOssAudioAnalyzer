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
import math
import platform

# Import Audio
import pyaudio
import librosa
import soundfile

import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt

# User Libraries
import pyOssWavfile					# 
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn

# Load Original Audio File

####################################
# 임펄스 음원
####################################

# imp_dir = 'ju_impulse'         # 임펄스 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'ju_impulse2'        # 2차년도 취득 임펄스 (32bit float / 48000Hz) 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'ju_impulse3'        # 3차년도 취득 임펄스 (32bit float / 48000Hz) 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'ju_impulse_test'    # 테스트 임펄스 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'ju_impulse2/ju_impulse2_original'	# 2차년도 취득 임펄스 (24bit int / 96kHz) 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'ju_impulse3/ju_impulse3_original'	# 3차년도 취득 임펄스 (24bit int / 96kHz) 음원 파일이 있는 프로젝트 내 폴더명
# imp_dir = 'GH_IR'		# 1차년도 취득 임펄스 중 금호아트홀 실감음향 임펄스(방향성)
imp_dir = 'reverb'		# 2차년도, 3차년도 취득 임펄스 중 실감음향 임펄스(방향성)

g_current_file = ''

for filename in os.listdir(imp_dir):
	if filename.endswith(".wav"):
		file_directory = os.path.join(imp_dir, filename)

		print("\n\n")
		print("file directory & name = ", file_directory)

		g_current_file = filename       #global

		fname = filename[:-4]

		# 임펄스 파일명 조합 
		imp_fname = pyOssWavfile.str_fname(imp_dir, fname)

		# 웨이브 파일 헤더 정보 확인 Check Header Information in Impulse Wav file 
	
		st_fmt_ori = pyOssWavfile.extractWavFmtChunk( pyOssWavfile.read_format(imp_fname) )
		dbg.dWavInfo(st_fmt_ori)
