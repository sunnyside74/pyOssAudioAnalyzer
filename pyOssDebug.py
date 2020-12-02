"""

"""

# Import Default
import os
import struct
import sys

# Import Library for math & plot
import numpy as np
import matplotlib.pyplot as plt

# Import User Library
import pyOssWavfile
import pyRoomAcoustic as room

#%matplotlib tk

def dPlotAudio(audio_fs, data_plot, title_txt="title", label_txt="label", xl_txt="x", yl_txt="y"):
	"""
    plot audio array using matplot for debug

    Parameters
    ----------
	audio_fs: samplig frequency of audio file from audio fromat chunk
    data_plot : data for plot
	title_txt : , optional
	label_txt : , optional
	xl_txt : , optional
	yl_txt : , optional

    Returns
    -------
	"""
	start_time = 0.
	end_time = data_plot.shape[0] / audio_fs	# fs: audio_fmt_chunk[3]

	plot_time = np.linspace(start_time, end_time, data_plot.shape[0])
	plt.title(title_txt)
	fig = plt.plot(plot_time, data_plot, label=label_txt)
	plt.legend()
	plt.xlabel(xl_txt)
	plt.ylabel(yl_txt)
	plt.xlim(0, end_time)
	plt.ylim(-1.0, 1.0)

	plt.show()


def dPlotDecay(audio_fs, data_plot, title_txt="title", label_txt="label", xl_txt="x", yl_txt="y"):
	"""
    plot audio array using matplot for debug

    Parameters
    ----------
	audio_fs: samplig frequency of audio file from audio fromat chunk
    data_plot : data for plot
	title_txt : , optional
	label_txt : , optional
	xl_txt : , optional
	yl_txt : , optional

    Returns
    -------
	"""
	start_time = 0.
	end_time = data_plot.shape[0] / audio_fs	# fs: audio_fmt_chunk[3]

	plot_time = np.linspace(start_time, end_time, data_plot.shape[0])
	plt.title(title_txt)
	fig = plt.plot(plot_time, data_plot, label=label_txt)
	plt.legend()
	plt.xlabel(xl_txt)
	plt.ylabel(yl_txt)
	plt.xlim(0, end_time)
	plt.ylim(-60, 0)
	
	plt.show()


def dPrint(func_name, dText, dData):
	"""
    Print function for Debug

    Parameters
    ----------
	func_name: function name text
    dText: Text for Display
	dData: Data for Display

    Returns
    -------
	"""
	print (func_name, dText, dData)


def dWavInfo(fname):
	"""
	Print Wave File Information

	Parameters
	----------
	fname: wave file path & name or struct_format_chunk
	Retruns
	--------
	None
	"""

	if type(fname) == str:		# when file name with path string
		struct_fmt = pyOssWavfile.extractWavFmtChunk(pyOssWavfile.read_format(fname))
	else:						# when fname is struct_fmt data
		struct_fmt = fname
	
	if struct_fmt.format == 1:
		str_format = 'Int'
	elif struct_fmt.format == 3:
		str_format = 'float'
	else:
		str_format = struct_fmt.format

	print("Audio Format =", str_format)
	print("Number of Channel =", struct_fmt.ch)
	print("Sampling Frequency =", struct_fmt.fs)
	# print("Byte Rate =", struct_fmt.byterate)       # 일종의 Checksum 
	# print("Block Align =", struct_fmt.blockalign)
	print("Bits per Sample =", struct_fmt.bitdepth)
	# print("Time =", time, "sec")
	# print("Length = ", data.shape[0])


def dAParam( data, decayCurveNorm, fs, RT60 = False, fname='' ):
	# Calculation Acoustic Parameters
	data_EDT, impulse_EDTnonLin = room.EDT(decayCurveNorm, fs)
	data_t20, impulse_t20nonLin = room.T20(decayCurveNorm, fs)
	data_t30, impulse_t30nonLin = room.T30(decayCurveNorm, fs)
	if RT60 is True:
		data_t60, impulse_t60nonLin = room.RT60(decayCurveNorm, fs) 
	else:
		data_t60 = data_t30
	data_D50 = room.D50(data, fs)
	data_C80 = room.C80(data, fs)
	data_C50 = room.C50(data, fs)

	print( "Impulse Name: ", fname)
	print( " - Decay Time  0 ~ -10dB =", data_EDT/6)	# for Debug
	print( " - Decay Time -5 ~ -25dB =", data_t20/3)	# for Debug
	print( " - Decay Time -5 ~ -35dB =", data_t30/2)	# for Debug
	print( " - EDT=", data_EDT)         				# for Debug
	print( " - T20=", data_t20)         				# for Debug
	print( " - T30=", data_t30)         				# for Debug
	if RT60 is True:
		print( " - RT60(Real)=", data_t60)				# for Debug
	else:
		print( " - RT60(from T30)=", data_t60) 			# for Debug
	print( " - D50=", data_D50)         				# for Debug
	print( " - C50=", data_C50)         				# for Debug
	print( " - C80=", data_C80)         				# for Debug




