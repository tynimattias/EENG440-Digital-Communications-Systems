import CommunicationLibLab9 as CL
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci



Str = 'This is EENG 440 Class!!! And EENG 440 is fun!!!'
BitStringArray = CL.text_to_bits(Str)

M = 8

BinBits = int(np.log2(M))

BinaryDataArray, BinaryNRZ, DecimalDataToBeTransmitted = CL.BinaryToDecimalArray(BitStringArray,BinBits)




# Dictionary for MPSK encoding
MPSKGrayEncode = {0:0,1:1,2:3,3:2,4:6,5:7,6:5,7:4}
FromOneToTwoBranch = {0:[1,0], 1:[1/np.sqrt(2),1/np.sqrt(2)],  
                      2:[0,1],3:[-1/np.sqrt(2),1/np.sqrt(2)],
                      4:[-1,0],5:[-1/np.sqrt(2),-1/np.sqrt(2)],
                      6:[0,-1], 7:[1/np.sqrt(2),-1/np.sqrt(2)]}

IchVal = np.zeros(len(DecimalDataToBeTransmitted))
QchVal = np.zeros(len(DecimalDataToBeTransmitted))


for n in range(len(DecimalDataToBeTransmitted)):
    GrayCode = MPSKGrayEncode[DecimalDataToBeTransmitted[n]]
    NBits = FromOneToTwoBranch[GrayCode]
    IchVal[n] = NBits[0]; QchVal[n] = NBits[1]

# Upsampling
UpsampledIchVal = CL.Upsampler(IchVal, 100); UpsampledQchVal = CL.Upsampler(QchVal, 100)

#Filter
h = np.ones(100)
h = h/np.linalg.norm(h)

FilteredIchVal = sci.convolve(UpsampledIchVal, h); FilteredQchVal = sci.convolve(UpsampledQchVal, h)

#carrier
Ts = 0.0002; fs = 1/Ts;
n = np.arange(0,len(FilteredIchVal),1)
t = n*Ts
fc=1600 # [Hz]

for n in range(len(FilteredIchVal)):
    ct = np.cos(2*np.pi*fc*t[n])
    ct_s = np.sin(2*np.pi*fc*t[n])

    FilteredIchVal[n] = FilteredIchVal[n]*ct
    FilteredQchVal[n] = FilteredQchVal[n]*ct_s

TransmitSignal = FilteredIchVal + FilteredQchVal

#Reciever

Ts = 0.0002; fs = 1/Ts;
n = np.arange(0,len(TransmitSignal),1)
t = n*Ts
fc=1600 # [Hz]

Recieved_I_Channel = np.zeros(len(TransmitSignal))
Recieved_Q_Channel = np.zeros(len(TransmitSignal))

for n in range(len(TransmitSignal)):
    ct = 2*np.cos(2*np.pi*fc*t[n])
    ct_s = 2*np.sin(2*np.pi*fc*t[n])

    Recieved_I_Channel[n] = TransmitSignal[n]*ct
    Recieved_Q_Channel[n] = TransmitSignal[n]*ct_s

FilteredIchVal_Recieved = sci.convolve(Recieved_I_Channel, h); FilteredQchVal_Recieved = sci.convolve(Recieved_Q_Channel, h)
FilteredIchVal_Recieved = FilteredIchVal_Recieved[len(h):len(h)+len(UpsampledIchVal)]
FilteredQchVal_Recieved = FilteredQchVal_Recieved[len(h):len(h)+len(UpsampledQchVal)]


#Downsampling
DownsampledIchVal = CL.Downsampler(FilteredIchVal_Recieved, 100); DownsampledQchVal = CL.Downsampler(FilteredQchVal_Recieved, 100)

MPSKBitLevel = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1]
IchDeModData, IchDeModStr = CL.MinDistDetector(DownsampledIchVal, MPSKBitLevel)
QchDeModData, QchDeModStr = CL.MinDistDetector(DownsampledQchVal, MPSKBitLevel)


    

FromTwoToOneBranch = np.array([[0,0,4,0,0],[0,5,0,3,0],[6,0,0,0,2],
          [0,7,0,1,0],[0,0,0,0,0]])


MPSKGrayDecode = {0:0,1:1,2:3,3:2,4:7,5:6,6:4,7:5}

#Decode data using the recovered Ich and Qch

TempBits = []; TempDeModDataStr = []; Greydecdata = np.zeros(len(DecimalDataToBeTransmitted))
for n in range(len(IchDeModData)):
    TempDec = FromTwoToOneBranch[int(IchDeModData[n])][int(QchDeModData[n])]
    DecodedGrayDecimal = MPSKGrayDecode[TempDec]
   
    NBitsStr = bin(DecodedGrayDecimal)[2:].zfill(BinBits)
    TempDeModDataStr.append(NBitsStr)
    for k in range(len(NBitsStr)):
        TempBits.append(int(NBitsStr[k]))
    DeModData = np.array(TempBits)
    DecodedDataBinartStr = ''.join(np.array(TempDeModDataStr, dtype=str))


recoveredStr = CL.text_from_bits(DecodedDataBinartStr)
print(recoveredStr)


    

