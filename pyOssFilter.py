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

# Import Signal Process
import numpy
import scipy.signal as signal
import matplotlib.pyplot as plt

# User Libraries
import pyRoomAcoustic as room       # room acoustic parameter
import pyOssDebug as dbg            # for debug


def band_range(fc, octave=0):
    """ Calculation Octave Band Range
    
    Parameters
    ------------
    fc: Center Frequency of bandpass Filter
    octave: 3: 1/3 Octave Band, Others: 1/1 Octave

    Returns
    ------------
    f1: low cutoff frequency of bandpass filter
    f2: high cutoff frequency of bandpass filter

    """
    if octave == 3:     # 1/3 octave band
        f1 = math.trunc(fc / (math.sqrt(2**(1/3))))
        f2 = math.trunc(fc * (math.sqrt(2**(1/3))))
    else:               # 1 octave band
        f1 = math.trunc(0.707 * fc)
        f2 = math.trunc(1.414 * fc)
    #print(f1, f2)

    return f1, f2



def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ butterworth bandpass filter Process
    
    Parameters
    ------------
    data: signal data for filtering
    lowcut: low cutoff frequency of bandpass filter
    highcut: High cutoff frequency of bandpass filter
    fs: Center Frequency of Band Pass Filter
    order: butterworth filter order

    Returns
    ------------
    filtered: filtered signal data 

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    if highcut >= nyq:
        #high = (nyq - 1) / nyq
        high = 0.99
    else:
        high = highcut / nyq

    print(high)
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filtered = signal.lfilter(b, a, data)
    return filtered



def calc_filt_impulse(in_data, fs, fc, filt_type='butt', order_tab=2, RT60=False, fname = "Please set file name"):
    """ Impulse 

    Parameters
    -----------
        in_data: input data array
        fs: sampling freq.
        fc: Center Freq. of Band Pass Filter
        filt_type: Filter Type
                'butt': Butterworth IIR Filter using bandpass_filter function(default)
                'fir': FIR Filter with Hamming window using scipy.signal.firwin & fftconvolve 
        order_tab:  On butterworth, filter's order (recommand order =< 4)
                    On fir, FIR filter's tab size
        RT60: Calculation real RT60 if set True (Not recommand) 
        fname: file name string of input data
    
    Returns
    ----------
        data_filtered: filtered data
        decaycurve: Normalized Decay Curve from Filtered Data 
        acoustic_param: array RT60, EDT, D50, C50, C80
    """
    if in_data.ndim != 1:
        data = in_data[:,0]
        str_ch_name = "Left Channel"
    else:
        data = in_data
        str_ch_name = "Mono"

    time = data.shape[0] / fs

    # Octave Band Pass Filter Range 
    band_f1, band_f2 = band_range(fc)

    if filt_type == 'butt':
        # Band Pass Filter Butterworth 2th order
        filter_name = "Butterworth 2nd Order" 
        data_filtered = bandpass_filter(data, band_f1, band_f2, fs, order=order_tab)
    elif filt_type == 'fir': 
        # Band Pass Filter FIR Hamming
        filter_name = "FIR" + str(order_tab) + "tab Hamming"
        firtab = order_tab
        if band_f2 > 20000:
            band_f2 = 20000
        coef_fir1 = numpy.float32(signal.firwin(firtab, [band_f1, band_f2], pass_zero=False, fs=fs))
        data_filtered = signal.fftconvolve(data, coef_fir1)

    # Plot Filtered Impulse Data
    dbg.dPlotAudio(fs, data_filtered, title_txt=fname+' filtered '+str(fc)+'Hz', label_txt=str_ch_name, xl_txt="Time(sec)", yl_txt= "Amplitude")

    # Calculation Normalized Decay Curve
    decaycurve = numpy.float32(room.decayCurve(data_filtered, time, fs))
    dbg.dPlotDecay(fs, decaycurve, title_txt=fname + ' decay curve ' + str(fc) + 'Hz', label_txt=str_ch_name, xl_txt="Time(sec)", yl_txt="Amplitude")

    # Calculation Acoustic Parameters
    data_EDT, impulse_EDTnonLin = room.EDT(decaycurve, fs)
    data_t20, impulse_t20nonLin = room.T20(decaycurve, fs)
    data_t30, impulse_t30nonLin = room.T30(decaycurve, fs)
    if RT60 is True:
        data_t60, impulse_t60nonLin = room.RT60(decaycurve, fs) 
    else:
        data_t60 = data_t30 * 2
    data_D50 = room.D50(data_filtered, fs)
    data_C80 = room.C80(data_filtered, fs)
    data_C50 = room.C50(data_filtered, fs)

    Cacoustic_param = CAcousticParameter(data_t60, data_EDT, data_D50, data_C50, data_C80)

    print("Impulse Name: " + fname + ", Filter: " + filter_name + ", " + str(fc) + "Hz" )
    print("T10=", data_EDT/6)      # for Debug
    print("T20=", data_t20)          # for Debug
    print("T30=", data_t30)          # for Debug
    if RT60 is True:
        print("RT60(Real)=", data_t60)            # for Debug
    else:
        print("RT60(from T30*2)=", data_t60)            # for Debug
    print("EDT=", data_EDT)            # for Debug
    print("D50=", data_D50)         # for Debug
    print("C50=", data_C50)         # for Debug
    print("C80=", data_C80)         # for Debug

    return  data_filtered, decaycurve, Cacoustic_param


def calc_filt_impulse_learning(draw_plot, in_data, fs, fc, filt_type='butt', order_tab=2, RT60=False, fname = "Please set file name"):
    """ Impulse

    Parameters
    -----------
        in_data: input data array
        fs: sampling freq.
        fc: Center Freq. of Band Pass Filter
        filt_type: Filter Type
                'butt': Butterworth IIR Filter using bandpass_filter function(default)
                'fir': FIR Filter with Hamming window using scipy.signal.firwin & fftconvolve
        order_tab:  On butterworth, filter's order (recommand order =< 4)
                    On fir, FIR filter's tab size
        RT60: Calculation real RT60 if set True (Not recommand)
        fname: file name string of input data

    Returns
    ----------
        data_filtered: Filtered data
        decaycurve: Normalized Decay Curve from Filtered Data
        acoustic_param: array RT60, EDT, D50, C50, C80
    """
    if in_data.ndim != 1:
        data = in_data[:,0]
        str_ch_name = "Left Channel"
    else:
        data = in_data
        str_ch_name = "Mono"

    time = data.shape[0] / fs

    # Octave Band Pass Filter Range
    band_f1, band_f2 = band_range(fc)

    if filt_type == 'butt':
        # Band Pass Filter Butterworth 2th order
        filter_name = "Butterworth 2nd Order"
        data_filtered = bandpass_filter(data, band_f1, band_f2, fs, order=order_tab)
    elif filt_type == 'fir':
        # Band Pass Filter FIR Hamming
        filter_name = "FIR" + str(order_tab) + "tab Hamming"
        firtab = order_tab
        if band_f2 > 20000:
            band_f2 = 20000
        coef_fir1 = numpy.float32(signal.firwin(firtab, [band_f1, band_f2], pass_zero=False, fs=fs))
        data_filtered = signal.fftconvolve(data, coef_fir1)

    # Plot Filtered Impulse Data
    if draw_plot:
        dbg.dPlotAudio(fs, data_filtered, fname + ' filtered ' + str(fc) + 'Hz', str_ch_name, "Time(sec)", "Amplitude")

    # Calculation Normalized Decay Curve
    decaycurve = numpy.float32(room.decayCurve(data_filtered, time, fs))

    # Plot DecayCurve
    if draw_plot:
        dbg.dPlotDecay( fs, decaycurve, fname + ' decay curve ' + str(fc) + 'Hz', str_ch_name, "Time(sec)", "Amplitude")


    # Calculation Acoustic Parameters
    data_EDT, impulse_EDTnonLin = room.EDT(decaycurve, fs)
    data_t20, impulse_t20nonLin = room.T20(decaycurve, fs)
    data_t30, impulse_t30nonLin, s_0dB, s_10dB, s_20dB, s_30dB, = room.T30_learning(decaycurve, fs)

    if RT60 is True:        # 현재 사용하지 않음  False
        data_t60, impulse_t60nonLin = room.RT60(decaycurve, fs)
    else:
        data_t60 = data_t30 * 2

    data_D50 = room.D50(data_filtered, fs)
    data_C80 = room.C80(data_filtered, fs)
    data_C50 = room.C50(data_filtered, fs)

    Cacoustic_param = CAcousticParameter(data_t60, data_EDT, data_D50, data_C50, data_C80)
    Csample_param   = CsampledBParameter(s_0dB, s_10dB, s_20dB, s_30dB)

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

    return  data_filtered, decaycurve, Cacoustic_param, Csample_param


'''
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta

# Several flavors of bandpass FIR filters.

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps

def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5*fs]
    taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
    return taps


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 63.0
    lowcut = 0.7
    highcut = 4.0

    ntaps = 128
    taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
    taps_kaiser16 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.6)
    taps_kaiser10 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)
    remez_width = 1.0
    taps_remez = bandpass_remez(ntaps, lowcut, highcut, fs=fs,
                                width=remez_width)

    # Plot the frequency responses of the filters.
    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((lowcut, 0), highcut - lowcut, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = freqz(taps_hamming, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Hamming window")

    w, h = freqz(taps_kaiser16, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.6")

    w, h = freqz(taps_kaiser10, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.0")

    w, h = freqz(taps_remez, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h),
             label="Remez algorithm, width=%.1f" % remez_width)

    plt.xlim(0, 8.0)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of several FIR filters, %d taps' % ntaps)

    plt.show()
    '''

class CAcousticParameter:
    def __init__(self, RT60, EDT, D50, C50, C80):
        self.RT60 = RT60

        self.EDT = EDT
        self.D50 = D50
        self.C50 = C50
        self.C80 = C80

class CsampledBParameter:
    def __init__(self, s_0dB, s_10dB, s_20dB, s_30dB):
        self.s_0dB = s_0dB
        self.s_10dB = s_10dB
        self.s_20dB = s_20dB
        self.s_30dB = s_30dB

