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



global g_current_file
g_current_file = ""

#################################
# Load Original Audio File
#################################

# Directories of Original Files
# ori_dir = 'impulsefiles'      # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (OpenAir 다운로드)
# ori_dir = 'ju_impulse2'       # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득 2차)
ori_dir = 'ju_impulse3'       # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득 3차)
# ori_dir = 'ju_anechoic3'      # 무향실 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득 3차)

# Directories of Converted Files
save_dir = 'ju_impulse3'
# save_dir = 'ju_anechoic3'

for filename in os.listdir(ori_dir):
    if filename.endswith(".wav"):
        file_directory = os.path.join(ori_dir, filename)

        print("\n\n\n")
        print("file_directory = ", file_directory)

        g_current_file = filename       #global

        fname = filename[:-4]
        ori_fname = pyOssWavfile.str_fname(ori_dir, fname)  # 파일명 조합

        # Check Original Impulse Wav file Header Information
        st_fmt_ori = pyOssWavfile.extractWavFmtChunk( pyOssWavfile.read_format(ori_fname) )
        dbg.dWavInfo(st_fmt_ori)

        # Load Original Audio & Convert format mono / float32 / 48000Hz 
        chunk_conv, data_conv, st_fmt_conv, t_conv = pyOssWavfile.readf32( ori_fname, samplerate=48000 )    # function of convert wav file to 32bit float format
        dbg.dWavInfo(st_fmt_conv)
        print(f" - Time(sec) =", t_conv)

        str_info_conv = pyOssWavfile.str_file_info(st_fmt_conv)
        dir_result = os.path.join(os.getcwd(), save_dir)
        pyOssWavfile.write(os.path.join(dir_result, fname + str_info_conv + '.wav'), st_fmt_conv.fs, data_conv)
