# Import Systems 
import struct
import io
import os
import sys
import math

import numpy
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt

# User Libraries
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter


def learning_decay(in_data, fs, tgt_rt=None, use_rt60=False):
	""" Impulse

	Parameters
	-----------
		in_data: input data array
		fs: sampling freq.
		RT60: Calculation real RT60 if set True (Not recommand)
		fname: file name string of input data

	Returns
	----------
		data: data
		decaycurve: Normalized Decay Curve from Filtered Data
		Cacoustic_param: structures RT60, EDT, D50, C50, C80
		Csample_param: structures x axis sample positon valeus of -10dB, -20dB, -30dB
	"""
	if in_data.ndim != 1:
		data = in_data[:,0]
		str_ch_name = "Left Channel"
	else:
		data = in_data
		str_ch_name = "Mono"

	# estimate_time = data.shape[0] / fs

	if tgt_rt is None:
		estimate_time = room.estimate_rt(data, fs)
	else:
		estimate_time = tgt_rt

	# Calculation Normalized Decay Curve
	decaycurve = numpy.float32(room.decayCurve(in_data, estimate_time, fs))

	# Calculation Acoustic Parameters
	data_EDT, impulse_EDTnonLin = room.EDT(decaycurve, fs)
	# data_t20, impulse_t20nonLin = room.T20(decaycurve, fs)
	data_t30, impulse_t30nonLin, s_0dB, s_10dB, s_20dB, s_30dB, = room.T30_learning(decaycurve, fs)

	if use_rt60 is True:        # 현재 사용하지 않음  False
		data_t60, impulse_t60nonLin = room.RT60(decaycurve, fs)
	else:
		data_t60 = data_t30 * 2

	data_D50 = room.D50(data, fs)
	data_C80 = room.C80(data, fs)
	data_C50 = room.C50(data, fs)

	Cacoustic_param = room.CAcousticParameter(data_t60, data_EDT, data_D50, data_C50, data_C80)
	Csample_param   = pyOssFilter.CsampledBParameter(s_0dB, s_10dB, s_20dB, s_30dB)

	return  data, decaycurve, Cacoustic_param, Csample_param


def calc_gain_slope(slope_VAL=0.15, RT=None, tgt_RT=None, CDecayCurvePos=None, valDataLength=None):
	
	if RT > tgt_RT:
		slope_a = numpy.ones( CDecayCurvePos.s_0dB, dtype='f' ) # 시작점(0dB)까지
		slope_b = numpy.logspace( 0, -slope_VAL, num=( CDecayCurvePos.s_10dB - CDecayCurvePos.s_0dB ) ) # 0dB ~ -10dB(EDT)
		slope_c = numpy.logspace( -slope_VAL, -slope_VAL-0.1, num=( CDecayCurvePos.s_30dB - CDecayCurvePos.s_10dB ) )# -10dB ~ -30dB 
		# slope_d = numpy.ones( ( valDataLength-CDecayCurvePos.s_30dB ), dtype='f' ) # (Reverberation)
		slope_d = numpy.logspace( slope_VAL-0.1, 0, num=( valDataLength-CDecayCurvePos.s_30dB ) )	# (Reverberation) 
	else:
		slope_a = numpy.ones( CDecayCurvePos.s_0dB, dtype='f' ) # 시작점(0dB)까지
		slope_b = numpy.logspace( 0, slope_VAL, num=( CDecayCurvePos.s_10dB - CDecayCurvePos.s_0dB ) ) # 0dB ~ -10dB(EDT)
		slope_c = numpy.logspace( slope_VAL, slope_VAL+0.1, num=( CDecayCurvePos.s_30dB - CDecayCurvePos.s_10dB ) )# -10dB ~ -30dB 
		# slope_d = numpy.ones( ( valDataLength-CDecayCurvePos.s_30dB ), dtype='f' ) # (Reverberation)
		slope_d = numpy.logspace( slope_VAL+0.1, 0, num=( valDataLength-CDecayCurvePos.s_30dB ) )	# (Reverberation) 

	slope = numpy.append( slope_a, slope_b)
	slope = numpy.append( slope, slope_c)
	slope = numpy.append( slope, slope_d)

	return slope
