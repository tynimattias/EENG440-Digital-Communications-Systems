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
    plt.pause(0.05)

def DrawScatterPlot(Ich,Qch,FigNo,Marker):
    plt.figure(FigNo)
    plt.plot(Ich,Qch,Marker)
    plt.grid(True)
    plt.pause(0.05)

def BinaryToDecimalArray(SizeOneStringArray,NumOfBits):
    # Disjoin size-1 string into binary 0/1 array
    BinaryDataArray = np.zeros(len(SizeOneStringArray),dtype=np.float64)
    BinaryNRZ = np.zeros(len(SizeOneStringArray),dtype=np.float64)
    for n in range(len(SizeOneStringArray)):
        if SizeOneStringArray[n] == '1':
            BinaryDataArray[n] = 1
            BinaryNRZ[n] = 1
        elif SizeOneStringArray[n] == '0':
            BinaryDataArray[n] = 0
            BinaryNRZ[n] = -1
        else:
            print('Error !!! Unknown binary string data!!!')
    # Change binary array to decimal numbers
    Iter = int(np.ceil(len(BinaryDataArray)/NumOfBits))
    TempBinaryData = np.zeros(Iter*NumOfBits)
    TempBinaryData[0:len(BinaryDataArray)] = BinaryDataArray
    DecNo = np.zeros(Iter)
    for n in range(Iter):
        BitsToBeDecimalNo = TempBinaryData[n*NumOfBits:(n+1)*NumOfBits]
        TempDec = 0
        for k in range(NumOfBits):
            if BitsToBeDecimalNo[k] == 1:
                TempDec = TempDec + 2**((NumOfBits-k)-1)
        DecNo[n] = TempDec
    return BinaryDataArray, BinaryNRZ, DecNo

def MinDistDetector(DownsampledSig,MPSKBitLevel):
    # Minimum distiance detector
    N = len(DownsampledSig); M = len(MPSKBitLevel)
    d = np.zeros((N,M))
    ds = np.zeros((N,M))
    for k in range(M):
        d[:,k] = (DownsampledSig - MPSKBitLevel[k]); ds[:,k] = np.multiply(d[:,k],d[:,k])

    DeModData = np.zeros(N)
    DeModDataStr = np.zeros(N,dtype=np.str)
      
    for n in range(N):
        DeModData[n] = np.argmin(ds[n,:])
        DeModDataStr[n] = np.str(DeModData[n])
        
    return DeModData,DeModDataStr

def GenerateBinaryBlockData(L):
    RandData = np.random.rand(L)
    BinaryStr = np.zeros(L, dtype=np.str)
    for n in range(L):
        if RandData[n] > 0.5:
            BinaryStr[n] = '1'
        else:
            BinaryStr[n] = '0'
    BinaryStrSizeOne = ''.join(BinaryStr)
    return BinaryStrSizeOne

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

