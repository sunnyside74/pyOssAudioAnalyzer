'''
# 강화학습 결과 테스트
- 임펄스 파일 불러오기
- 불러온 임펄스 파일을 이용하여 강화학습 수행
- 강화학습 결과물의 음향 파라미터를 연산하고 wav파일로 저장
- 
-  
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
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn

#############################################################################
#  Impulse Data Load & -------------------------------------------------------#
#############################################################################

str_fileinfo = '_mono_32f_44.1k'    # 파일명에 부가된 정보

# LOAD IMPULSE WAVE FILE
# OpenAir 임펄스 파일 
imp_dir = 'impulsefiles'       # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (OpenAir 다운로드)

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


# 2차년도 취득 임펄스 파일
# imp_dir = 'ju_impulse'         # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득)

# imp_name = '경기국악당 IR-01.mono.32f.48k'
# imp_name = '국립국악원 우면당 IR-01.mono.32f.48k'
# imp_name = '김해문화의전당 IR.mono.32f.48k'
# imp_name = '김해서부문화센터 IR.mono.32f.48k'
# imp_name = '노원문화예술회관 IR.mono.32f.48k'
# imp_name = '대구범어성당 IR.mono.32f.48k'
# imp_name = '돈화문국악당.mono.32f.48k'
# imp_name = '세종문화회관 대극장 IR-01.mono.32f.48k'
# imp_name = '세종문화회관 채임버홀.mono.32f.48k'
# imp_name = '승동교회 IR.mono.32f.48k'
# imp_name = '아람누리 음악당 IR.mono.32f.48k'
# imp_name = '아람콘서트홀 IR.mono.32f.48k'
# imp_name = '애터미공연장 IR-01.mono.32f.48k'
# imp_name = '우란문화재단 공연장 IR-01.mono.32f.48k'
# imp_name = '울산중구문화의전당 IR-01.mono.32f.48k'
# imp_name = '전주완산여고 IR.mono.32f.48k'
# imp_name = '천안예술의전당 IR-01.mono.32f.48k'
# imp_name = '청주아트홀 IR.mono.32f.48k'
# imp_name = '풍류홀 IR.mono.32f.48k'
# imp_name = '한국문화의집 IR.mono.32f.48k'

imp_fname= imp_name + str_fileinfo


impulse_fname = pyOssWavfile.str_fname(imp_dir, imp_fname) # 임펄스 파일에 대한 전체경로, 파일명 조합
fmt_imp, data_imp, st_fmt_imp, t_imp = pyOssWavfile.readf32(impulse_fname)
print(t_imp)

# 3초짜리 데이터로 만듬 (3초보다 짧은 임펄스 파일에 무음을 뒤에 추가하여 3초짜리 데이터로 만듬)
if t_imp < 3.0:
    t_temp = 3.0 - t_imp
    data = pyOssWavfile.insertSilence(data_imp, st_fmt_imp.fs, t_temp)
    print(data.shape[0]/st_fmt_imp.fs)


###############################################################################
# Reinforcement Learing
###############################################################################




###############################################################################
# Test Audio Data(Anechoic Audio) Load
###############################################################################
# Load audio
aud_dir = 'audiofiles'
aud_name = "singing"
aud_fname= aud_name + str_fileinfo

audio_fname = pyOssWavfile.str_fname(aud_dir, aud_fname) # 전체경로, 파일명 조합
fmt_aud, data_aud, st_fmt_aud, t_a = pyOssWavfile.readf32(audio_fname)


###############################################################################
# Filter Process
###############################################################################

fc = 500        # Center freq for bandpass filter 500Hz

# Impulse
data_filt, decay, a_param, c_param = pyOssFilter.calc_filt_impulse_learning(False, data_imp, st_fmt_imp.fs, fc, fname=impulse_fname)
print(a_param.__dict__)
print(c_param.__dict__)



# Convolution Process with Anechoic audio data and Filtered impulse data
data_convolve_ori = sig.fftconvolve(data_aud, data_filt)

# Save Convolved Wav File
sname_ori = pyOssWavfile.str_fname('', aud_name + '_ori_' + imp_name + str_fileinfo)
print(sname_ori)
pyOssWavfile.write(sname_ori, st_fmt_imp.fs, data_convolve_ori)
print('* Save complete convolution data original')


###############################################################################
#
###############################################################################
tgt_rt60 = 2.5      # unit: sec
sample_tgt_rt60 = c_param.s_0dB + int(st_fmt_imp.fs * tgt_rt60)
print(c_param.s_0dB, sample_tgt_rt60)


k = 1
draw_plot = False

if a_param.RT60[0][0] > tgt_rt60:
    print("... > ", str(tgt_rt60))
    while a_param.RT60[0][0] > tgt_rt60:

        # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
        # 위치 찾기
        p_0dB = c_param.s_0dB
        p_10dB = c_param.s_10dB
        p_20dB = c_param.s_20dB
        p_30dB = c_param.s_30dB

        '''
        #case 1
        data_filt = data_filt * 0.8
        data_temp = data_filt

        # if k == 26:
        #     draw_plot = True
        # else:
        #     draw_plot = False

        # data_w_filtered, decaycurve_w_filtered, acoustic_w_param, sample_w_dB_param  = \
        #     pyOssFilter.calc_filt_impulse_learning(draw_plot, data_w, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname + str_fileinfo)
        data_filt, decay, a_param, c_param  = \
            learn.learning_decay(draw_plot, data_temp, st_fmt_w.fs, fc, fname=wav_fname)
        '''
        '''
        #case 2
        if p_10dB > p_0dB and p_30dB > 0:
            data_filt[0:p_10dB] = data_filt[0:p_10dB] * 0.9

            if p_20dB > p_10dB:
                data_filt[p_10dB:] = data_filt[p_10dB:] * 0.7

            data_temp = data_filt

            # if k == 26:
            #     draw_plot = True
            # else:
            #     draw_plot = False

            # data_filt, decay, a_param, c_param  = \
            #     pyOssFilter.calc_filt_impulse_learning(draw_plot, data_temp, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname)
            data_filt, decay, a_param, c_param  = \
                learn.learning_decay(draw_plot, data_temp, st_fmt_w.fs, fc, fname=wav_fname)
        '''
        '''
        #case 3
        if p_10dB > p_0dB and p_30dB > 0:
            data_w_filtered[0:p_10dB] = data_w_filtered[0:p_10dB] * 0.8

            if p_20dB > p_10dB:
                data_w_filtered[p_10dB:p_20dB] = data_w_filtered[p_10dB:p_20dB] * 0.5

            if p_30dB > p_20dB:
                data_w_filtered[p_20dB:p_30dB] = data_w_filtered[p_20dB:p_30dB] * 0.3

            data_w = data_w_filtered
            if k == 26:
                draw_plot = True
            else:
                draw_plot = False

            data_w_filtered, decaycurve_w_filtered, acoustic_w_param, sample_w_dB_param  = \
                pyOssFilter.calc_filt_impulse_learning(draw_plot, data_w, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname + str_fileinfo)
        '''
        #case 4 (oss)
        gain_slope_a = np.ones(p_0dB, dtype='f')
        print( len(gain_slope_a) )
        # gain_slope_b = np.linspace( 1.0, 0.7, num=(data_filt.shape[0]-p_0dB) )
        gain_slope_b = np.logspace( 0, -0.1, num=(data_filt.shape[0]-p_0dB) )
        gain_slope = np.append( gain_slope_a, gain_slope_b )
        data_temp = data_filt * gain_slope
        data_filt, decay, a_param, c_param  = \
            learn.learning_decay(draw_plot, data_temp, st_fmt_imp.fs, fc, fname=imp_fname)

        if a_param.RT60[0][0] == 0.0 or k > 1000:
            break

        k = k + 1
        # if k <= 50 or k % 50 == 0:
            # print (k, " : ",  a_param.RT60[0][0])
            # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
        print (k, " : ",  a_param.RT60[0][0])
        print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
else:
    print("... < ", str(tgt_rt60))
    while a_param.RT60[0][0] < tgt_rt60:

        # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
        # 위치 찾기
        p_0dB = c_param.s_0dB
        p_10dB = c_param.s_10dB
        p_20dB = c_param.s_20dB
        p_30dB = c_param.s_30dB

        '''
        #case 1
        data_filt = data_filt * 1.2
        data_temp = data_filt

        # if k == 50:
        #     draw_plot = True
        # else:
        #     draw_plot = False

        # data_w_filtered, decaycurve_w_filtered, acoustic_w_param, sample_w_dB_param  = \
        #     pyOssFilter.calc_filt_impulse_learning(draw_plot, data_w, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname + str_fileinfo)
        data_filt, decay, a_param, c_param  = \
            learn.learning_decay(draw_plot, data_temp, st_fmt_w.fs, fc, fname=wav_fname)
        '''
        '''
        #case 2
        if p_10dB > p_0dB and p_30dB > 0:
            data_filt[0:p_10dB] = data_filt[0:p_10dB] * 1.0

            if p_20dB > p_10dB:
                data_filt[p_10dB:] = data_filt[p_10dB:] * 1.1

            data_temp = data_filt

            # if k == 29:   
            #     draw_plot = True
            # else:
            #     draw_plot = False

            # data_filt, decay, a_param, c_param  = \
            #     pyOssFilter.calc_filt_impulse_learning(draw_plot, data_temp, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname + str_fileinfo)
            data_filt, decay, a_param, c_param  = \
                learn.learning_decay(draw_plot, data_temp, st_fmt_w.fs, fc, fname=wav_fname)
        '''
        '''
        #case 3
        if p_10dB > p_0dB and p_30dB > 0:
            data_w_filtered[0:p_10dB] = data_w_filtered[0:p_10dB] * 1.1

            if p_20dB > p_10dB:
                data_w_filtered[p_10dB:p_20dB] = data_w_filtered[p_10dB:p_20dB] * 1.2

            if p_30dB > p_20dB:
                data_w_filtered[p_20dB:p_30dB] = data_w_filtered[p_20dB:p_30dB] * 1.5

            data_w = data_w_filtered

            if k == 26:
                draw_plot = True
            else:
                draw_plot = False

            data_w_filtered, decaycurve_w_filtered, acoustic_w_param, sample_w_dB_param  = \
                pyOssFilter.calc_filt_impulse_learning(draw_plot, data_w, st_fmt_w.fs, fc, filt_type='butt',fname=wav_fname + str_fileinfo)
        '''

        #case 4 
        gain_slope_a = np.ones(p_0dB, dtype='f')
        # gain_slope_b = np.linspace( 1.0, 1.3, num=(data_filt.shape[0]-p_0dB) )
        gain_slope_b = np.logspace( 0, 0.1, num=(data_filt.shape[0]-p_0dB) )
        gain_slope = np.append( gain_slope_a, gain_slope_b )
        data_temp = data_filt * gain_slope
        data_filt, decay, a_param, c_param  = \
            learn.learning_decay(draw_plot, data_temp, st_fmt_w.fs, fc, fname=wav_fname)

        if a_param.RT60[0][0] == 0.0 or k > 1000:
            print("K IS ==== ", k)
            break

        k = k + 1
        # if k <= 50 or k % 50 == 0:
            # print (k, " : ",  a_param.RT60[0][0])
            # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
        print (k, " : ",  a_param.RT60[0][0])
        print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)

print("=== Stop, k = ", k)
# print("2-1. acoustic_w_param = ", acoustic_w_param)
# print('1. inspect = ', inspect.getmembers(acoustic_w_param))
print('\n2-1. __dict__ = ', a_param.__dict__)
print('2-2. acoustic_w_param.RT60[0][0] = ', a_param.RT60[0][0])
print('\n2-3. __dict__ = ', c_param.__dict__)
print('2-4. sample_w_dB_param.s_0dB = ', c_param.s_0dB)
print('2-5. sample_w_dB_param.s_10dB = ', c_param.s_10dB)
print('2-6. sample_w_dB_param.s_20dB = ', c_param.s_20dB)
print('2-7. sample_w_dB_param.s_30dB = ', c_param.s_30dB)

dbg.dPlotAudio(st_fmt_w.fs, gain_slope)

dbg.dPlotAudio( st_fmt_w.fs, data_filt, fname + ' filtered ' + str(fc) + 'Hz', label_txt='k='+str(k), xl_txt="Time(sec)", yl_txt="Amplitude" )
dbg.dPlotDecay( st_fmt_w.fs, decay, fname + ' decay curve ' + str(fc) + 'Hz', label_txt='k='+str(k), xl_txt="Time(sec)", yl_txt="Amplitude" )

# Convolution
data_convolve_trans = sig.fftconvolve(data_a, data_filt)

# Save Wav File
sname_trans = pyOssWavfile.str_fname('', aud_name + '_trans_' + wav_name + str_fileinfo)
print(sname_trans)
pyOssWavfile.write(sname_trans, st_fmt_w.fs, data_convolve_trans)
print('* Save complete convolution data trans')


'''
##########
# pyOssTest.py 참조
##########
# 임펄스 응담 계산 후 원음과 컨볼루션(fftconvolve)하여 들어보기


# Audio File Load
dir_audio = 'anechoic_sample'
str_info = '_mono_32f_44.1k'

aud_name = 'singing'
aud_name_temp = 'singing' + str_info
aud_fname = pyOssWavfile.str_fname(dir_audio, aud_name_temp)
print(aud_name_temp)

chunk_aud, data_aud, st_fmt_aud, t_aud = pyOssWavfile.readf32(aud_fname)
print('* Load complete audio data')

# Impulse File Load
dir_imp = 'impulse_sample'

# imp_name = 'ju_imp_goyang_aramnuri_concerthall'
# imp_name = 'ju_imp_sejongmunhwahuigwan_chamberhall'
imp_name = 'mh3_000_ortf_48k'
#imp_name = "MaesHowe"

imp_name_temp = imp_name + str_info
imp_fname = pyOssWavfile.str_fname(dir_imp, imp_name_temp)
print(imp_name_temp)

chunk_imp, data_imp, st_fmt_imp, t_imp = pyOssWavfile.readf32(imp_fname)
print('* Load complete impulse data')

# FFT Convolution Function Test
# Test Audio data with Impulse data

data_convolve_temp = sig.fftconvolve(data_aud, data_w_filtered)
# print(data_aud.shape[0], st_fmt_aud.fs, data_aud.shape[0]+st_fmt_aud.fs)
# data_convolve = data_convolve_temp[0:(data_aud.shape[0]+st_fmt_aud.fs)]     # audio data time + 1 sec (it's not full length of convolution)
data_convolve = data_convolve_temp

# dbg.dPlotAudio(st_fmt_imp.fs, data_imp, imp_name_temp, "Mono", "Time(sec)", "Amplitude")  # plot Impulse
# dbg.dPlotAudio(st_fmt_aud.fs, data_aud, aud_name + ' original', "Mono", "Time(sec)", "Amplitude")   # plot original audio
# dbg.dPlotAudio(st_fmt_aud.fs, data_convolve, aud_name + ' convolve with ' + 'impulse', "Mono", "Time(sec)", "Amplitude")    # plot Convolved Audio

# Save wav file Convolution Result

sname = pyOssWavfile.str_fname('', aud_name + '_' + imp_name + str_info)
print(sname)
pyOssWavfile.write(sname, st_fmt_aud.fs, data_convolve)
print('* Save complete convolution data')
'''
