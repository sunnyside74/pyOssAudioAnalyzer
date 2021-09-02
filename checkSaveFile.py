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

imp_dir = 'resultfiles'         # 임펄스 음원 파일이 있는 프로젝트 내 폴더명

imp_name = '경기국악당 IR-01.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'
imp_name = '.mono.32f.48k'

# 임펄스 파일명 조합
imp_fname = pyOssWavfile.str_fname(imp_dir, imp_name)

# Check Original Impulse Wav file Header Information
st_fmt_ori = pyOssWavfile.extractWavFmtChunk( pyOssWavfile.read_format(imp_fname) )
dbg.dWavInfo(st_fmt_ori)

