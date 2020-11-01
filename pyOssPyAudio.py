# Import Systems 
import struct
import io
import os
import sys
import math
import platform

# Import Audio
import pyaudio
# import sounddevice
# import librosa
# import soundfile

import numpy as np
# import scipy
# import scipy.signal as sig
# import matplotlib.pyplot as plt

# User Libraries
# import pyOssWavfile
# import pyRoomAcoustic as room
# import pyOssDebug as dbg
# import pyOssFilter


def ju_get_device_name(val_index):
    """ Jeonju Univ. 원하는 HOST API의 디바이스 정보 얻기 in pyaudio

    Parameters
    ----------
        val_index: Host API type index값 입력 (MME, ASIO 등)
        device_name: 디바이스 이름 string 값

    Returns
    -------
        device_name: (dict type)

    """
    device_name = []        # 빈 리스트 생성

    host_api = pyaudio.PyAudio().get_host_api_info_by_type(val_index)
    api_index = host_api['index']
    device_cnt = pyaudio.PyAudio().get_device_count()

    for i in range(0, device_cnt):
        temp = pyaudio.PyAudio().get_device_info_by_index(i)
        if temp['hostApi'] == api_index:
            device_name.append(temp['name'])
        
    if device_name is None:
        raise ValueError('No Matching Host API')

    return (device_name)


def ju_get_device_info(val_index, device_name):
    """ Jeonju Univ. 원하는 HOST API의 디바이스 정보 얻기 in pyaudio

    Parameters
    ----------
        val_index: Host API type index값 입력 (MME, ASIO 등)
        device_name: 디바이스 이름 string 값

    Returns
    -------
        device_info: 디바이스 정보 (dict type)

    """
    device_info = {}

    host_api = pyaudio.PyAudio().get_host_api_info_by_type(val_index)
    api_index = host_api['index']
    device_cnt = pyaudio.PyAudio().get_device_count()

    for i in range(0, device_cnt):
        temp = pyaudio.PyAudio().get_device_info_by_index(i)
        if (temp['hostApi'] == api_index) and (temp['name'] == device_name):
            device_info = temp

    if not device_info:
        raise ValueError('No Matching device name')

    return (device_info)


# def ju_audio_callback(in_data, frame_count, time_info, status):
#     data = in_data 


class CAudioDeviceInfo:
    def __init__(self, index, name, hostApi, maxInputCh, maxOutputCh, fs):
        self.index = index
        self.name = name
        self.hostApi = hostApi
        self.maxInCh = maxInputCh
        self.maxOutCh = maxOutputCh
        self.fs = int(fs)

