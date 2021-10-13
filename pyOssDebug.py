"""

"""

# Import Default
import struct
import io
import os
import sys
import math
import platform


# Import Library for math & plot
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import False_

# Import User Library
import pyOssWavfile
import pyRoomAcoustic as room

#%matplotlib tk

def dPlotAudio(audio_fs, data_plot, y_range=1.0, title_txt="title", label_txt="label", xl_txt="x", yl_txt="y", newWindow=False):
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

	if newWindow == True:
		plt.figure()
	plt.title(title_txt)
	fig = plt.plot(plot_time, data_plot, label=label_txt)
	plt.legend()
	plt.xlabel(xl_txt)
	plt.ylabel(yl_txt)
	plt.xlim(0, end_time)
	plt.ylim(-y_range, y_range)

	plt.show()


def dPlotDecay(audio_fs, data_plot, title_txt="title", label_txt="label", xl_txt="x", yl_txt="y", newWindow=False):
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

	if newWindow == True:
		plt.figure()
	plt.title(title_txt)
	fig = plt.plot(plot_time, data_plot, label=label_txt)
	plt.legend()
	plt.xlabel(xl_txt)
	plt.ylabel(yl_txt)
	plt.xlim(0, end_time)
	plt.ylim(-60, 0)
	
	plt.show()


def dPrintFunc(func_name, dText, dData):
	"""
    Print function for Debug

    Parameters
    ----------
	func_name: string of function name text
    dText: string of Text for Display
	dData: string of Data for Display

    Returns
    -------
	"""
	print (func_name, dText, dData)


def dPrintf(strings):
	"""
	Print Input strings for Debug

	Parameters
	----------
	strings: text strings
	"""
	print (strings)


def dWavInfo(fname):
	"""	Print Wave File Information

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

	print(" - Audio Format =", str_format)
	print(" - Number of Channel =", struct_fmt.ch)
	print(" - Sampling Frequency =", struct_fmt.fs)
	# print(" - Byte Rate =", struct_fmt.byterate)       # 일종의 Checksum 
	# print(" - Block Align =", struct_fmt.blockalign)
	print(" - Bits per Sample =", struct_fmt.bitdepth)
	# print(" - Time =", time, "sec")
	# print(" - Length = ", data.shape[0])


def dPrintAParam(CAcousticParam):
	""" Print Acoustic Parameters in structure of pyRoomAcoustic.CAcousticParameters
	:param CAcousticParam: structure of Acoustic Paramter
	"""
	print( "Stucture of the Acoustic Parameters")
	print( " - RT60 = ", CAcousticParam.RT60[0][0])				# for Debug
	print( " - EDT = ", CAcousticParam.EDT[0][0])				# for Debug
	print( " - D50 = ", CAcousticParam.D50)         			# for Debug
	print( " - C50 = ", CAcousticParam.C50)         			# for Debug
	print( " - C80 = ", CAcousticParam.C80)         			# for Debug

	print( " - Decay Time  0 ~ -10dB = ", CAcousticParam.EDT[0][0]/6)	# for Debug
	# print( " - Decay Time -5 ~ -25dB = ", CAcousticParam.T20[0][0]/3)	# for Debug
	print( " - Decay Time -5 ~ -35dB = ", CAcousticParam.RT60[0][0]/4)	# for Debug
	# print( " - T20 = ", CAcousticParam.T20[0][0])         			# for Debug
	# print( " - T30 = ", CAcousticParam.T30[0][0]/2)         			# for Debug

