# -*- coding: utf-8 -*-
"""
This file is to have all required functions for labs of
EENG 440 (Digital Communication Systems).

Programmed by Min-sung Koh
"""
import numpy as np
import matplotlib.pyplot as plt
import binascii

def DrawSpectrum(Sig,fs,color):
    Len = len(Sig) # N = Len
    if Sig.shape[0] == 1:
        Sig = Sig.T
        
    Windows = np.hamming(Len)
    Len = int(2**(np.ceil(np.log2(Len))+2))
    # Normalize 0~2pi in w to be nomalized freq 0~1.
    # Take out -pi~pi and map it into -0.5~0.5 in f
    # Hence, the freq more than 0.5 in f is wrapped.
    f = ((np.arange(0,Len,1)/Len)-0.5)*fs
    Spec = np.fft.fftshift(np.abs(np.fft.fft(Sig*Windows,Len)),0)
    plt.plot(f,Spec,color); plt.grid(True)

def Upsampler(x,M):
    z = np.zeros(M*len(x))
    i=0
    for nVal in range(len(z)):
        if (nVal % M)==0:
            z[nVal]=x[i]
            i += 1
    return z

def Downsampler(x,M):
    z = np.zeros(int(np.ceil(len(x)/M)))
    i=0
    for nVal in range(len(x)):
        if (nVal % M)==0:
            z[i]=x[nVal]
            i += 1
    return z
    
def DrawEyeDiagram(Data,NumOfDataInBlock,FigNo):
    plt.figure(FigNo)
    Indx = np.arange(0,NumOfDataInBlock)
    NumOfBlocks = int(np.floor(len(Data)/NumOfDataInBlock))
    for k in range(NumOfBlocks):
        plt.plot(Indx,Data[k*NumOfDataInBlock:(k+1)*NumOfDataInBlock],'b')
       
    plt.grid()

def DrawScatterPlot(Ich,Qch,FigNo,Marker):
    plt.figure(FigNo)
    plt.plot(Ich,Qch,Marker)
    plt.grid()
    
# "text_to_bits", "text_from_bits", and "int2bytes" are from
# https://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
# and those are little revised for this lab.
def text_to_bits(text, encoding='ascii', errors='ignore'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='ascii', errors='ignore'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))
