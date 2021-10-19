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
#  Impulse Data Load & 
#############################################################################

# str_fileinfo = '_mono_32f_44.1k'    # 파일명에 부가된 정보

STAT_FILTER = False                 # Filter Process Off: False, On: True
STAT_SAVE_IMPULSE = True            # 필터처리 또는 강화학습 한 임펄스 데이터 저장 여부
STAT_SAVE_WAV_ORI = False          # 음장 처리된 파일의 저장 여부 선택
STAT_SAVE_RESULT = False           # 임펄스 적용 결과물 저장 여부 선택 No Save: False, Save: True
STAT_DRAW = True                   # 결과물 그림 출력
STAT_SAVE_FIGURE = True            # 결과물 이미지 저장 여부

result_dir = 'resultfiles'         # 결과물을 저장할 경로

dbg.dPrintf( 'STAT_FILTER = ' + str( STAT_FILTER ) )
dbg.dPrintf( 'STAT_SAVE_IMPULSE = ' + str( STAT_SAVE_IMPULSE ) )
dbg.dPrintf( 'STAT_SAVE_WAV_ORI = ' + str( STAT_SAVE_WAV_ORI ) )
dbg.dPrintf( 'STAT_SAVE_RESULT = ' + str( STAT_SAVE_RESULT ) )
dbg.dPrintf( 'STAT_DRAW = ' + str( STAT_DRAW ) )


# LOAD IMPULSE WAVE FILE
# OpenAir 임펄스 파일 
# imp_dir = 'impulsefiles'       # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (OpenAir 다운로드)

# imp_name = "ElvedenHallMarbleHall.mono.32f.48k"
# imp_name = "EmptyApartmentBedroom.mono.32f.48k"
# imp_name = "FalklandPalaceRoyalTennisCourt.mono.32f.48k"
# imp_name = "InsidePiano.mono.32f.48k"
# imp_name = "MaesHowe.mono.32f.48k"
# imp_name = "SportsCentreUniversityOfYork.mono.32f.48k"
# imp_name = "StairwayUniversityOfYork.mono.32f.48k"
# imp_name = "StAndrewsChurch.mono.32f.48k"
# imp_name = "mh3_000_ortf_48k.mono.32f.48k"
# imp_name = "TyndallBruceMonument.mono.32f.48k"


# 2차년도 취득 임펄스 파일
imp_dir = 'ju_impulse'         # 임펄스 음원 파일이 있는 프로젝트 내 폴더명 (전주대, 사운드코리아이엔지 직접 취득)

# imp_name = '경기국악당 IR-01.mono.32f.48k'
imp_name = '국립국악원 우면당 IR-01.mono.32f.48k'
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

imp_fname= imp_name

impulse_fname = pyOssWavfile.str_fname(imp_dir, imp_fname) # 임펄스 파일에 대한 전체경로, 파일명 조합
fmt_imp, data_imp, st_fmt_imp, t_imp = pyOssWavfile.readf32(impulse_fname, 48000)
dbg.dPrintf( 't_imp =' + str(t_imp) )

fs = st_fmt_imp.fs   # Set Samplerate frequency

# 3초짜리 데이터로 만듬 (3초보다 짧은 임펄스 파일에 무음을 뒤에 추가하여 3초짜리 데이터로 만듬)
if t_imp < 3.0:
    t_temp = 3.0 - t_imp
    data = pyOssWavfile.insertSilence(data_imp, fs, t_temp)
    dbg.dPrintf(data.shape[0]/fs)

# 불러온 임펄스 파일의 음향 파라미터 특성 출력
decay_imp = room.decayCurve(data_imp, estimate=None, fs=fs)
imp_param = room.calcAcousticParam(data_imp, decay_imp, fs)
dbg.dPrintAParam(imp_param)
# dbg.dPrintAParam(room.calcAcousticParam(data_imp, room.decayCurve(data_imp, estimate=None, fs=fs), fs))
if STAT_DRAW:
    dbg.dPlotDecay(fs, decay_imp, imp_name, imp_name)

###############################################################################
# Test Audio Data(Anechoic Audio) Load
# - 불러오는 오디오 파일의 샘플링주파수, 비트가 다르다면 readf32 메소드 내에서 
#   임펄스 파일의 주파수와 비트와 같도록 리샘플링 처리 및 비트를 변환하여 
#   임펄스 파일과 동일하게 함
###############################################################################
# Load anechoic audio file
aud_dir = 'audiofiles'
aud_name = "singing"
aud_fname= aud_name

audio_fname = pyOssWavfile.str_fname(aud_dir, aud_fname) # 전체경로, 파일명 조합
fmt_aud, data_aud, st_fmt_aud, t_aud = pyOssWavfile.readf32(audio_fname, fs)    # 파일 불러오기 (조건에 따라 변환도 수행)

###############################################################################
# Filter Process to loaded impulse data and save filtered impulse
###############################################################################
if STAT_FILTER == True:         # 로드한 임펄스를 필터링하고, 필터링 된 임펄스로 오디오 처리, 임펄스 강화학습 처리
    # '''
    array_fc = [500, 1000, 2000, 4000, 8000, 16000]     # Octave Band filter

    for fc in array_fc:
        dbg.dPrintf( '\nfc = ' + str(fc) )
        data_filt, decay, a_param, c_param = pyOssFilter.calc_filt_impulse_learning(False, data_imp, fs, fc, fname=impulse_fname)
        dbg.dPrintf( a_param.__dict__ )
        dbg.dPrintf( c_param.__dict__ )

        # Save filtering impulse data
        imp_filt_fname = imp_name + '.filtered_' + str(fc) + 'Hz'

        if STAT_SAVE_IMPULSE == True:
            sname_imp_filt = pyOssWavfile.str_fname(result_dir, imp_filt_fname)
            # dbg.dPrintf(sname_imp_filt)  # for debug
            pyOssWavfile.write(sname_imp_filt, fs, data_filt)
            print('* Save complete %s', imp_filt_fname)

        # Convolution Process with Anechoic audio data and Filtered impulse data
        data_convolve_ori = sig.fftconvolve(data_aud, data_filt)
        ori_name = imp_filt_fname

        if STAT_SAVE_WAV_ORI == True:
            # Save Wav File convolve filtered Audio
            sname_ori = pyOssWavfile.str_fname(result_dir, aud_name + '.ori.' + ori_name) # 파일경로 + 파일이름
            # dbg.dPrintf(sname_ori)  # for debug
            pyOssWavfile.write(sname_ori, fs, data_convolve_ori)    # 무향실 음원에 필터링 된 임펄스를 적용한 wav file 저장
            print('* Save complete %s', aud_name + '.ori.' + ori_name)

        # Reinforcement Learning Process with filtered impulse data
        data_learn = data_filt          # 강화학습에 사용할 임펄스 데이터는 '필터 처리 한 임펄스 데이터'
        trans_name = imp_filt_fname     # 강화학습 처리 한 음장처리 결과 파일 저장에 사용할 이름

        # Reinforcement Learning
        tgt_rt60 = 2.5      # sec
        sample_tgt_rt60 = c_param.s_0dB + int(fs * tgt_rt60)
        print('Pos 0dB = %d, Tgt RT60 = %d' % (c_param.s_0dB, sample_tgt_rt60))

        k = 1
        draw_plot = False
        # draw_plot = True

        if a_param.RT60[0][0] > tgt_rt60:
            print("... > ", str(tgt_rt60))
            while a_param.RT60[0][0] > tgt_rt60:

            # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
            # 위치 찾기
                # p_0dB = c_param.s_0dB
                # p_10dB = c_param.s_10dB
                # p_20dB = c_param.s_20dB
                # p_30dB = c_param.s_30dB

                #case 4: 
                # gain_slope_a = np.ones(p_0dB, dtype='f')
                # print( len(gain_slope_a) )
                # # gain_slope_b = np.linspace( 1.0, 0.7, num=(data_learn.shape[0]-p_0dB) )
                # gain_slope_b = np.logspace( 0, -0.1, num=(data_learn.shape[0]-p_0dB) )
                # gain_slope = np.append( gain_slope_a, gain_slope_b )

                # case 5
                # gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
                # gain_slope_b = np.logspace( 0, -0.1, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
                # gain_slope_c = np.logspace( 0, -0.2, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB 
                # gain_slope_d = np.ones((data_learn.shape[0]-p_30dB), dtype='f') # (Reverberation)
                # gain_slope = np.append( gain_slope_a, gain_slope_b)
                # gain_slope = np.append( gain_slope, gain_slope_c)
                # gain_slope = np.append( gain_slope, gain_slope_d)

                # case 5-1
                gain_slope = learn.calc_gain_slope(a_param.RT60[0][0], tgt_rt60, c_param, data_learn.shape[0])

                # Process
                data_temp = data_learn * gain_slope
                data_learn, decay, a_param, c_param  = \
                    learn.learning_decay(draw_plot, data_temp, fs, fc, fname=imp_fname)

                if a_param.RT60[0][0] == 0.0 or k > 1000:
                    print("K IS ==== ", k)
                    break

                k = k + 1
                # if k <= 50 or k % 50 == 0:
                    # print (k, " : ",  a_param.RT60[0][0])
                    # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
                # print (k, " : ",  a_param.RT60[0][0])
                # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
        else:
            print("... < ", str(tgt_rt60))
            while a_param.RT60[0][0] < tgt_rt60:

                # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
                # 위치 찾기
                # p_0dB = c_param.s_0dB
                # p_10dB = c_param.s_10dB
                # p_20dB = c_param.s_20dB
                # p_30dB = c_param.s_30dB

                #case 4 
                # gain_slope_a = np.ones(p_0dB, dtype='f')
                # # gain_slope_b = np.linspace( 1.0, 1.3, num=(data_filt.shape[0]-p_0dB) )
                # gain_slope_b = np.logspace( 0, 0.1, num=(data_learn.shape[0]-p_0dB) )
                # gain_slope = np.append( gain_slope_a, gain_slope_b )

                # case 5
                # gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
                # gain_slope_b = np.logspace( 0, 0.1, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
                # gain_slope_c = np.logspace( 0, 0.2, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB 
                # gain_slope_d = np.ones((data_learn.shape[0]-p_30dB), dtype='f') # (Reverberation)
                # gain_slope = np.append( gain_slope_a, gain_slope_b)
                # gain_slope = np.append( gain_slope, gain_slope_c)
                # gain_slope = np.append( gain_slope, gain_slope_d)

                # case 5-1
                gain_slope = learn.calc_gain_slope(a_param.RT60[0][0], tgt_rt60, c_param, data_learn.shape[0])

                data_temp = data_learn * gain_slope
                data_learn, decay, a_param, c_param  = \
                    learn.learning_decay(draw_plot, data_temp, fs, fc, fname=imp_fname)

                if a_param.RT60[0][0] == 0.0 or k > 1000:
                    print("K IS ==== ", k)
                    break

                k = k + 1
                # if k <= 50 or k % 50 == 0:
                    # print (k, " : ",  a_param.RT60[0][0])
                    # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
                # print (k, " : ",  a_param.RT60[0][0])
                # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)

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

        data_convolve_trans = sig.fftconvolve(data_aud, data_learn)

        # Save Learned Impulse & Processed Anechoic Audio file
        if STAT_SAVE_IMPULSE == True:
            imp_learn_fname = imp_name + '_' + str(imp_param.RT60[0][0]) + '_to_' + str(a_param.RT60[0][0]) + '_k=' + str(k)
            sname_imp_learn = pyOssWavfile.str_fname(result_dir, imp_learn_fname)
            # dbg.dPrintf(sname_imp_learn)
            pyOssWavfile.write(sname_imp_learn, fs, data_learn)
            dbg.dPrintf('* Save complete learned impulse data')

        if STAT_SAVE_RESULT == True:
            # Save learning processed with anecoic wav file
            sname_trans = pyOssWavfile.str_fname(result_dir, aud_name + '.trans.' + trans_name)
            # dbg.dPrintf(sname_trans)
            pyOssWavfile.write(sname_trans, fs, data_convolve_trans)
            dbg.dPrintf('* Save complete convolution data trans')
else:
    data_filt, decay, a_param, c_param = pyOssFilter.calc_filt_impulse_learning(False, data_imp, fs, fc=0, fname=impulse_fname)
    data_convolve_ori = sig.fftconvolve(data_aud, data_imp)
    ori_name = imp_fname

    if STAT_SAVE_WAV_ORI == True:
        sname_ori = pyOssWavfile.str_fname(result_dir, aud_name + '.ori.' + ori_name) # 파일경로 + 파일이름
        # dbg.dPrintf(sname_ori)  # for debug
        pyOssWavfile.write(sname_ori, fs, data_convolve_ori)    # 무향실 음원에 필터링 된 임펄스를 적용한 wav file 저장
        print('* Save complete %s',ori_name)

    data_learn = data_imp           # 사용할 임펄스 데이터가 원본 임펄스 데이터
    trans_name = imp_fname          # 강화학습 처리 한 음장처리 결과 파일 저장에 사용할 이름          

    # Reinforcement Learning
    tgt_rt60 = 2.5      # sec
    sample_tgt_rt60 = c_param.s_0dB + int(fs * tgt_rt60)
    print('Pos 0dB = %d, Tgt RT60 = %d' % (c_param.s_0dB, sample_tgt_rt60))

    k = 1
    draw_plot = False
    # draw_plot = True

    if a_param.RT60[0][0] > tgt_rt60:
        print("... > ", str(tgt_rt60))
        while a_param.RT60[0][0] > tgt_rt60:

            # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
            # 위치 찾기
            # p_0dB = c_param.s_0dB
            # p_10dB = c_param.s_10dB
            # p_20dB = c_param.s_20dB
            # p_30dB = c_param.s_30dB

            #case 4: 
            # gain_slope_a = np.ones(p_0dB, dtype='f')
            # print( len(gain_slope_a) )
            # # gain_slope_b = np.linspace( 1.0, 0.7, num=(data_learn.shape[0]-p_0dB) )
            # gain_slope_b = np.logspace( 0, -0.1, num=(data_learn.shape[0]-p_0dB) )
            # gain_slope = np.append( gain_slope_a, gain_slope_b )

            # case 5
            # gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
            # gain_slope_b = np.logspace( 0, -0.1, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
            # gain_slope_c = np.logspace( 0, -0.2, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB 
            # gain_slope_d = np.ones((data_learn.shape[0]-p_30dB), dtype='f') # (Reverberation)
            # gain_slope = np.append( gain_slope_a, gain_slope_b)
            # gain_slope = np.append( gain_slope, gain_slope_c)
            # gain_slope = np.append( gain_slope, gain_slope_d)

            # case 5-1
            gain_slope = learn.calc_gain_slope(a_param.RT60[0][0], tgt_rt60, c_param, data_learn.shape[0])

            # Process
            data_temp = data_learn * gain_slope
            data_learn, decay, a_param, c_param  = \
                learn.learning_decay(draw_plot, data_temp, fs, fc=0, fname=imp_fname)

            if a_param.RT60[0][0] == 0.0 or k > 1000:
                break

            k = k + 1
            # if k <= 50 or k % 50 == 0:
                # print (k, " : ",  a_param.RT60[0][0])
                # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
            # print (k, " : ",  a_param.RT60[0][0])
            # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
    else:
        print("... < ", str(tgt_rt60))
        while a_param.RT60[0][0] < tgt_rt60:

            # data_w2 각 구간별 위치 구한 후 각 구각에 data_w*1.4, *1.2 *1 계산
            # 위치 찾기
            # p_0dB = c_param.s_0dB
            # p_10dB = c_param.s_10dB
            # p_20dB = c_param.s_20dB
            # p_30dB = c_param.s_30dB

            #case 4 
            # gain_slope_a = np.ones(p_0dB, dtype='f')
            # # gain_slope_b = np.linspace( 1.0, 1.3, num=(data_filt.shape[0]-p_0dB) )
            # gain_slope_b = np.logspace( 0, 0.1, num=(data_learn.shape[0]-p_0dB) )
            # gain_slope = np.append( gain_slope_a, gain_slope_b )

            # case 5
            # gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
            # gain_slope_b = np.logspace( 0, 0.1, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
            # gain_slope_c = np.logspace( 0, 0.2, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB 
            # gain_slope_d = np.ones((data_learn.shape[0]-p_30dB), dtype='f') # (Reverberation)
            # gain_slope = np.append( gain_slope_a, gain_slope_b)
            # gain_slope = np.append( gain_slope, gain_slope_c)
            # gain_slope = np.append( gain_slope, gain_slope_d)

            # case 5-1
            gain_slope = learn.calc_gain_slope(a_param.RT60[0][0], tgt_rt60, c_param, data_learn.shape[0])

            data_temp = data_learn * gain_slope
            data_learn, decay, a_param, c_param  = \
                learn.learning_decay(draw_plot, data_temp, fs, fc=0, fname=imp_fname)

            if a_param.RT60[0][0] == 0.0 or k > 1000:
                print("K IS ==== ", k)
                break

            k = k + 1
            # if k <= 50 or k % 50 == 0:
                # print (k, " : ",  a_param.RT60[0][0])
                # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)
            # print (k, " : ",  a_param.RT60[0][0])
            # print ("      ",  p_0dB, p_10dB, p_20dB, p_30dB)

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

    # Convolution Anechoic Audio with Reinforcement Learned Impulse data
    data_convolve_trans = sig.fftconvolve(data_aud, data_learn)

    if STAT_SAVE_IMPULSE == True:
        # Save Learned Impulse
        imp_learn_fname = imp_name + '_' + str(imp_param.RT60[0][0]) + '_to_' + str(a_param.RT60[0][0]) + '_k=' + str(k)
        sname_imp_learn = pyOssWavfile.str_fname(result_dir, imp_learn_fname)
        # dbg.dPrintf(sname_imp_learn)
        pyOssWavfile.write(sname_imp_learn, fs, data_learn)
        dbg.dPrintf('* Save complete learned impulse data')

    if STAT_SAVE_RESULT == True:
        # Save learning processed with anecoic wav file
        sname_trans = pyOssWavfile.str_fname(result_dir, aud_name + '.trans.' + trans_name)
        # dbg.dPrintf(sname_trans)
        pyOssWavfile.write(sname_trans, fs, data_convolve_trans)
        dbg.dPrintf('* Save complete convolution data trans')


# if STAT_DRAW:
#     dbg.dPlotAudio(fs, gain_slope)

if STAT_DRAW:
    dbg.dPlotAudio(fs, data_learn, title_txt=trans_name, label_txt='k='+str(k), xl_txt='Time(sec)', yl_txt='Amplitude' )

if STAT_DRAW:
    dbg.dPlotDecay(fs, decay, ' decay curve of ' + trans_name, label_txt='k='+str(k), xl_txt='Time(sec)', yl_txt='Amplitude' )

