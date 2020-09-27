"""
Module to calcurate acoustic parameter using NumPy arrays

from https://
"""

import numpy as np
import scipy.stats as stats
from scipy import signal


def T20(decayCurveNorm, fs):
    """Calculate T20

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T20
    """
    T, nonLin = _reverberation(decayCurveNorm, -5, -25, fs)
    T20 = T * 3
    return T20, nonLin


def T30(decayCurveNorm, fs):
    """Calculate T30

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T30
    """

    T, nonLin = _reverberation(decayCurveNorm, -5, -35, fs)
    T30 = T * 2
    return T30, nonLin


def T60(decayCurveNorm, fs):
    """Calculate T60

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T60
    """

    T, nonLin = _reverberation(decayCurveNorm, -5, -65, fs)
    T60 = T
    return T60, nonLin


def EDT(decayCurveNorm, fs):
    """Calculate Early Decay Time (EDT)

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: EDT
    """

    T, nonLin = _reverberation(decayCurveNorm, 0, -10, fs)
    EDT = T * 6
    return EDT, nonLin


def C50(IR, fs):
    """Calculate clarity for speech (C50)

    :param IR: impulse response
    :param fs: sample rate
    :return: C50
    """

    C50 = _clarity(IR, 50, fs)
    return C50


def C80(IR, fs):
    """Calculate clarity for music (C80)

    :param IR: impulse response
    :param fs: sample rate
    :return: C80
    """

    C80 = _clarity(IR, 80, fs)
    return C80


def D50(IR, fs):
    """Calculate definition (D50)

    :param IR: impulse response
    :param fs: sample rate
    :return: D50
    """

    D50 = _definition(IR, 50, fs)
    return D50


def centreTime(IR, fs):
    """Calculate the centre time from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :return: centre time
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]

    t = np.repeat(np.linspace(0, np.size(IR, axis=0) / fs, np.size(IR, axis=0))[:, np.newaxis], np.size(IR, axis=1), axis=1)
    Ts = np.divide(np.sum(t * IR ** 2, axis=0), np.sum(IR ** 2, axis=0))
    if len(Ts) == 1:
        Ts = float(Ts)
    if Ts.ndim == 1:
        Ts = Ts[:, np.newaxis]
    return Ts


def lateralEnergyFraction(IR, IROmni, fs):
    """Calculate the lateral energy fraction from two impuleses

    :param IR: is an impulse from a figure of eight microphone
    :param IROmni: is an impulse from an omnidirection microphone
    :param fs: sample rate
    :return: lateral energy fraction
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]
    if IROmni.ndim == 1:
        IROmni = IROmni[:, np.newaxis]
    LF = np.divide(np.sum(IR[np.int64(5 / 1000 * fs):np.int64(80 / 1000 * fs)] ** 2, axis=0), np.sum(IROmni[np.int64(0 / 1000 * fs):np.int64(80 / 1000 * fs)] ** 2, axis=0))
    if LF.ndim == 1:
        LF = LF[:, np.newaxis]
    return LF


def strength(IR, IRSpeaker):
    """Calculate the strength from impulse responses

    :param IR: impulse response
    :param IRSpeaker: impulse response of the loudspeaker at 10 m in free field
    :return: strength
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]
    if IRSpeaker.ndim == 1:
        IRSpeaker = IRSpeaker[:, np.newaxis]

    G = 10 * np.log10(np.sum(IR ** 2, axis=0) / np.sum(IRSpeaker ** 2, axis=0))
    return G


#### HELP FUNCTIONS ####

def decayCurve(sig, estimate, fs, noiseEnd=0):
    """Calculate the decay curve from a noise signal

    :param sig: noise signal
    :param estimate: the estimated reverb time (s)
    :param fs: sample rate
    :param noiseEnd: the time at which the noise stimuli stops
    :return:
    """

    decayCurvePa = exponential(sig, estimate / 40, fs)
    decayCurveSPL = 20 * np.log10(abs(decayCurvePa))
    decayCurveNorm = decayCurveSPL - np.max(decayCurveSPL[int(noiseEnd * fs):], axis=0)
    return decayCurveNorm


def exponential(S, tau, fs):
    T = 1 / fs
    b = np.array([T / (T + tau)])
    b2 = np.array([1, -tau / (T + tau)])

    y = signal.lfilter(b, b2, S ** 2, axis=0)
    Ptau = np.sqrt(y)
    return Ptau


def _reverberation(decayCurveNorm, fs, reqDBStart=-5, reqDBEnd=-60):
    """Calculate reverberation based on requirements for start and stop level

    :param decayCurveNorm: normalized decay curve
    :param fs: sample rate
    :param reqDBStart: start level for reverberation (is -5 for T60 and 0 for EDT)
    :param reqDBEnd: end level for reverberation (is -60 for T60)
    :return: reveration
    """

    if decayCurveNorm.ndim == 1:
        decayCurveNorm = decayCurveNorm[:, np.newaxis]

    # create return arrays based on number of decay curves
    T = np.zeros((np.size(decayCurveNorm, axis=1), 1))
    nonLinearity = np.zeros((np.size(decayCurveNorm, axis=1), 1))

    for i in range(np.size(decayCurveNorm, axis=1)):
        try:
            sample0dB = np.where(decayCurveNorm[:, i] < reqDBStart)[0][0]  # find first sample below 0 dB
        except IndexError:
            raise ValueError("The is no level below {} dB".format(reqDBStart))

        try:
            sample10dB = sample0dB + np.where(decayCurveNorm[:, i][sample0dB:] <= reqDBEnd)[0][0]  # find first sample below -10dB
        except IndexError:
            raise ValueError("The is no level below required {} dB".format(reqDBEnd))

        testDecay = decayCurveNorm[:, i][sample0dB:sample10dB]  # slice decaycurve to specific samples
        print(testDecay.shape)  #for Debug

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(sample0dB / fs, sample10dB / fs, np.size(testDecay, axis=0)), testDecay)  # calculate the slope and of signal nonlinearity
        nonLinearity[i] = np.round(1000 * (1 - r_value ** 2), 1)  # calculate the nonlinearity
        x = np.arange(-5, len(decayCurveNorm[:, i]) / fs + 5, 1 / 1000)  # create straight line with 1 ms resolution +- 5 secons
        y = intercept + slope * x  # generate line with slop of calculated slope

        T[i] = len(y[np.where(y <= reqDBStart)[0][0]:np.where(y <= reqDBEnd)[0][0]]) / 1000  # calculate reverberation with 1 ms resolution from generated linear line
    return T, nonLinearity


def _clarity(IR, fs, t=50):
    """Calculate the clarity from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :param t: is the defined shift from early to late reflections (is often 50 ms or 80 ms)(ms)
    :return: clarity
    """

    C = 10 * np.log10(np.sum(IR[0:np.int64((t / 1000) * fs)] ** 2, axis=0) / np.sum(IR[np.int64((t / 1000) * fs):] ** 2, axis=0))
    if C.ndim == 1:
        C = C[:, np.newaxis]
    return C


def _definition(IR, fs, t=50):
    """Calculate the defintion from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :param t: is the defined shift from early to late reflections (is often 50 ms)(ms)
    :return: definition
    """

    D = np.sum(IR[0:np.int64(t / 1000 * fs)] ** 2, axis=0) / np.sum(IR ** 2, axis=0)
    if D.ndim == 1:
        D = D[:, np.newaxis]
    return D
