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
	plt.plot(plot_time, data_plot, label=label_txt)
	plt.legend()
	plt.xlabel(xl_txt)
	plt.ylabel(yl_txt)
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
