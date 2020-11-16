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


def learning_decay(draw_plot, in_data, fs, fc, RT60=False, fname = "Please set file name"):
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

    t_time = data.shape[0] / fs

    # Plot Filtered Impulse Data
    if draw_plot:
        dbg.dPlotAudio(fs, data, fname + ' filtered ' + str(fc) + 'Hz', str_ch_name, "Time(sec)", "Amplitude")

    # Calculation Normalized Decay Curve
    decaycurve = numpy.float32(room.decayCurve(in_data, t_time, fs))

    # Plot DecayCurve
    if draw_plot:
        dbg.dPlotDecay( fs, decaycurve, fname + ' decay curve ' + str(fc) + 'Hz', str_ch_name, "Time(sec)", "Amplitude")

    # Calculation Acoustic Parameters
    data_EDT, impulse_EDTnonLin = room.EDT(decaycurve, fs)
    # data_t20, impulse_t20nonLin = room.T20(decaycurve, fs)
    data_t30, impulse_t30nonLin, s_0dB, s_10dB, s_20dB, s_30dB, = room.T30_learning(decaycurve, fs)

    if RT60 is True:        # 현재 사용하지 않음  False
        data_t60, impulse_t60nonLin = room.RT60(decaycurve, fs)
    else:
        data_t60 = data_t30 * 2

    data_D50 = room.D50(data, fs)
    data_C80 = room.C80(data, fs)
    data_C50 = room.C50(data, fs)

    Cacoustic_param = pyOssFilter.CAcousticParameter(data_t60, data_EDT, data_D50, data_C50, data_C80)
    Csample_param   = pyOssFilter.CsampledBParameter(s_0dB, s_10dB, s_20dB, s_30dB)

    # for DEBUG
    # print("Impulse Name: " + fname + ", Filter: " + filter_name + ", " + str(fc) + "Hz" )
    # print("T10=", data_EDT/6)      # for Debug
    # print("T20=", data_t20)          # for Debug
    # print("T30=", data_t30)          # for Debug
    # if RT60 is True:
    #     print("RT60(Real)=", data_t60)          # for Debug
    # else:
    #     print("RT60(from T30*2)=", data_t60)    # for Debug
    # print("EDT=", data_EDT)         # for Debug
    # print("D50=", data_D50)         # for Debug
    # print("C50=", data_C50)         # for Debug
    # print("C80=", data_C80)         # for Debug

    # print("Start   0dB=", Csample_param.s_0dB)
    # print("Start -10dB=", Csample_param.s_10dB)
    # print("Start -20dB=", Csample_param.s_20dB)
    # print("Start -30dB=", Csample_param.s_30dB)

    return  data, decaycurve, Cacoustic_param, Csample_param
