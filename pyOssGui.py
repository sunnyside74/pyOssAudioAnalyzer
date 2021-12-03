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

# Import GUI
import tkinter as tk
from tkinter import filedialog

# User Libraries


def ossTkFileOpen(pathname=os.getcwd(), title="Choose your file"):
    init_dir = pathname

    filename = tk.filedialog.askopenfilename(initialdir = init_dir,
                                             title = title, 
                                             filetypes = (("wave files","*.wav"),("npz files","*.npz"),("all files","*.*")))

    print(filename)
    return filename

def ossTkFileSave():
    init_dir = os.getcwd()
    title = "Save your file"

    filename = tk.filedialog.asksavefilename(initialdir = init_dir,
                                             title = title, 
                                             filetypes = (("wave files","*.wav"),("npz files","*.npz"),("all files","*.*")))
    return filename

