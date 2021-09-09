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
imp_dir = 'impulsefiles'       # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (OpenAir 다운로드)
# imp_dir = 'ju_impulse'         # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득)

imp_name = "ElvedenHallMarbleHall"
# imp_name = "EmptyApartmentBedroom"
# imp_name = "FalklandPalaceRoyalTennisCourt"
# imp_name = "InsidePiano"
# imp_name = "MaesHowe"
# imp_name = "SportsCentreUniversityOfYork"
# imp_name = "StairwayUniversityOfYork"
# imp_name = "StAndrewsChurch"
# imp_name = "mh3_000_ortf_48k"
# imp_name = "TyndallBruceMonument"

# imp_name = '경기국악당 IR-01.mono.24i.96k'
# imp_name = '국립국악원 우면당 IR-01.mono.24i.96k'
# imp_name = '김해문화의전당 IR.mono.24i.96k'
# imp_name = '김해서부문화센터 IR.mono.24i.96k'
# imp_name = '노원문화예술회관 IR.mono.24i.96k'
# imp_name = '대구범어성당 IR.mono.24i.96k'
# imp_name = '돈화문국악당.mono.24i.96k'
# imp_name = '세종문화회관 대극장 IR-01.mono.24i.96k'
# imp_name = '세종문화회관 채임버홀.mono.24i.96k'
# imp_name = '승동교회 IR.mono.24i.96k'
# imp_name = '아람누리 음악당 IR.mono.24i.96k'
# imp_name = '아람콘서트홀 IR.mono.24i.96k'
# imp_name = '애터미공연장 IR-01.mono.24i.96k'
# imp_name = '우란문화재단 공연장 IR-01.mono.24i.96k'
# imp_name = '울산중구문화의전당 IR-01.mono.24i.96k'
# imp_name = '전주완산여고 IR.mono.24i.96k'
# imp_name = '천안예술의전당 IR-01.mono.24i.96k'
# imp_name = '청주아트홀 IR.mono.24i.96k'
# imp_name = '풍류홀 IR.mono.24i.96k'
# imp_name = '한국문화의집 IR.mono.24i.96k'

# 임펄스 파일명 조합
imp_fname = pyOssWavfile.str_fname(imp_dir, imp_name)

# Check Original Impulse Wav file Header Information
st_fmt_ori = pyOssWavfile.extractWavFmtChunk( pyOssWavfile.read_format(imp_fname) )
dbg.dWavInfo(st_fmt_ori)

############################################################################################
# Load Original Audio & Convert format mono / float32 / 48000Hz 
############################################################################################
chunk_conv, data_conv, st_fmt_conv, t_conv = pyOssWavfile.readf32( imp_fname, samplerate=48000 )    # function of convert wav file to 32bit float format
dbg.dWavInfo(st_fmt_conv)
print(f" - Time(sec) =", t_conv)

############################################################################################
# Save coverted date as .wav and .npz file
############################################################################################
str_info_conv = pyOssWavfile.str_file_info(st_fmt_conv)

# Set saving directory 
dir_result = os.path.join(os.getcwd(), imp_dir)
# save .wav file
# pyOssWavfile.write(os.path.join(dir_result, imp_name + str_info_conv + '.wav'), st_fmt_conv.fs, data_conv)
# save npz file
# pyOssWavfile.save_oss_npz(os.path.join(dir_result, imp_name + str_info_conv + '.npz'), data_conv, st_fmt_conv, t_conv)
