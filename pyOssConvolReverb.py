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
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn


STAT_OVERLAP_ADD = False                # Option of Overlap-Add Convolution 

STAT_SAVE_RESULT = True                 # Filter Process Off: False, On: True
# STAT_SAVE_RESULT = False                 # Filter Process Off: False, On: True


fs = 48000

###############################################################################
# Test Audio Data(Anechoic Audio) Load
# - 불러오는 오디오 파일의 샘플링주파수, 비트가 다르다면 readf32 메소드 내에서 
#   임펄스 파일의 주파수와 비트와 같도록 리샘플링 처리 및 비트를 변환하여 
#   임펄스 파일과 동일하게 함
###############################################################################
# Load anechoic audio file
aud_dir = 'ju_anechoic3'

imp_dir = 'ju_impulse3'

save_dir = 'ju_result_reverb'         # 결과물을 저장할 경로

g_current_audio_file = ""
g_current_impulse_file = ""


for audioFilename in os.listdir(aud_dir):
    if audioFilename.endswith(".wav"):
        audio_file_directory = os.path.join(aud_dir, audioFilename)

        print("\n\n\n")
        print("audio_file_directory = ", audio_file_directory)

        g_current_audio_file = audioFilename       #global

        aud_name = audioFilename[:-4]

        aud_fname = pyOssWavfile.str_fname(aud_dir, aud_name) # 전체경로, 파일명 조합
        fmt_aud, data_aud, st_fmt_aud, t_aud = pyOssWavfile.readf32(aud_fname, fs)    # 파일 불러오기 (조건에 따라 변환도 수행)
        dbg.dWavInfo(st_fmt_aud)
        print(f" - Time(sec) =", t_aud)

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

                # decayNorm = room.decayCurve(sig=data_imp, estimate=None, fs=fs)
                # rt60 = room.T30(decayNorm, fs)
                # print("   RT60 = ", rt60[0][0])

                data_convolve = sig.fftconvolve(data_aud, data_imp)

                if STAT_SAVE_RESULT:
                    sname = pyOssWavfile.str_fname(save_dir, aud_name + '_' + imp_name) # 파일경로 + 파일이름
                    pyOssWavfile.write(sname, fs, pyOssWavfile.normalize(data_convolve))    # 무향실 음원에 필터링 된 임펄스를 적용한 wav file 저장
                    # pyOssWavfile.write(sname, fs, pyOssWavfile.normalize(data_convolve[:int((t_aud+rt60[0][0])*fs)]))    # 무향실 음원에 필터링 된 임펄스를 적용한 wav file 저장
                    print('* Save complete Convolution Reverb')






