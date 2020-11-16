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
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn

#############################################################################
#  Impulse Data Load & -------------------------------------------------------#
#############################################################################

str_fileinfo = '_mono_f32_44.1k'    # 파일명에 부가된 정보

# IMPULSE WAVE FILE
# wav_fname = "TyndallBruceMonument"
# wav_fname = "ElvedenHallMarbleHall'"
# wav_fname = "EmptyApartmentBedroom"
# wav_fname = "FalklandPalaceRoyalTennisCourt"
# wav_fname = "InsidePiano"
# wav_fname = "MaesHowe"
# wav_fname = "SportsCentreUniversityOfYork"
# wav_fname = "StairwayUniversityOfYork"
# wav_fname = "StAndrewsChurch"
wav_fname = "mh3_000_ortf_48k"
# wav_fname = "mh3_000_wx_48k"
# wav_fname = "anechoic_gunshot_0.44.1kHz.f32"
# wav_fname = "10s sweep for 3s reverb 48k"

dir_name = 'impulsefiles'

wav_fname= wav_fname + str_fileinfo

fname = pyOssWavfile.str_fname(dir_name, wav_fname) # 전체경로, 파일명 조합

fmt_w, data_w, st_fmt_w, t_w = pyOssWavfile.readf32(fname)
print(t_w)

if t_w < 3.0:
    t_temp = 3.0 - t_w
    data = pyOssWavfile.insertSilence(data_w, st_fmt_w.fs, t_temp)
    print(data.shape[0]/st_fmt_w.fs)


###############################################################################
# Filter
###############################################################################

fc = 500        # Center freq for bandpass filter 500Hz

data_filt, decay, a_param, c_param = pyOssFilter.calc_filt_impulse_learning(False, data, st_fmt_w.fs, fc, fname=wav_fname)
print(c_param.__dict__)

###############################################################################
#
###############################################################################
tgt_rt60 = 1.6      # sec
sample_tgt_rt60 = st_fmt_w.fs

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

dbg.dPlotAudio( st_fmt_w.fs, data_filt, fname + ' filtered ' + str(fc) + 'Hz', label_txt='k='+str(k), xl_txt="Time(sec)", yl_txt="Amplitude" )
dbg.dPlotDecay( st_fmt_w.fs, decay, fname + ' decay curve ' + str(fc) + 'Hz', label_txt='k='+str(k), xl_txt="Time(sec)", yl_txt="Amplitude" )


'''
##########
# pyOssTest.py 참조
##########
# 임펄스 응담 계산 후 원음과 컨볼루션(fftconvolve)하여 들어보기


# Audio File Load
dir_audio = 'anechoic_sample'
str_info = '_mono_f32_44.1k'

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

