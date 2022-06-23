from Comm_Systems import *
import numpy as np
import matplotlib.pyplot as plt
import MQAMLib as mqam


Str = 'This is EENG 440 Class!!! And EENG 440 is fun!!!'
BitStringArray = CL.text_to_bits(Str)

M = 8

Binbits = int(np.log2(M))

BinaryDataArray, BinaryNRZ, DecimalDataToBeTransmitted = CL.BinaryToDecimalArray(BitStringArray,Binbits)


# Dictionary for MPSK encoding
MPSKGrayEncode = {0:0,1:1,2:3,3:2,4:6,5:7,6:5,7:4}
FromOneToTwoBranch = {0:[1,0], 1:[1/np.sqrt(2),1/np.sqrt(2)],  
                      2:[0,1],3:[-1/np.sqrt(2),1/np.sqrt(2)],
                      4:[-1,0],5:[-1/np.sqrt(2),-1/np.sqrt(2)],
                      6:[0,-1], 7:[1/np.sqrt(2),-1/np.sqrt(2)]}

I_channel = BPSK(100, phase = 0)
Q_channel = BPSK(100, phase = np.pi/2)

IchVal = np.zeros(len(DecimalDataToBeTransmitted)); QchVal = np.zeros(len(DecimalDataToBeTransmitted))

for n in range(len(DecimalDataToBeTransmitted)):
    GrayCode = MPSKGrayEncode[DecimalDataToBeTransmitted[n]]
    NBits = FromOneToTwoBranch[GrayCode]
    IchVal[n] = NBits[0]; QchVal[n] = NBits[1]

I_channel_data = I_channel.transmitter(1500, IchVal)
Q_channel_data = Q_channel.transmitter(1500, QchVal)

transmitted_signal = I_channel_data + Q_channel_data

I_channel_recovered = I_channel.demodulator(transmitted_signal)
Q_channel_recovered = Q_channel.demodulator(transmitted_signal)


MPSKBitLevel = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1]

IchDeModData, IchDeModStr = I_channel.MinDistDetector(MPSKBitLevel)
QchDeModData, QchDeModStr = Q_channel.MinDistDetector(MPSKBitLevel)

FromTwoToOneBranch = np.array([[0,0,4,0,0],[0,5,0,3,0],[6,0,0,0,2],
          [0,7,0,1,0],[0,0,0,0,0]])


MPSKGrayDecode = {0:0,1:1,2:3,3:2,4:7,5:6,6:4,7:5}

#Decode data using the recovered Ich and Qch

TempBits = []; TempDeModDataStr = []; Greydecdata = np.zeros(len(DecimalDataToBeTransmitted))
for n in range(len(IchDeModData)):
    TempDec = FromTwoToOneBranch[int(IchDeModData[n])][int(QchDeModData[n])]
    DecodedGrayDecimal = MPSKGrayDecode[TempDec]
   
    NBitsStr = bin(DecodedGrayDecimal)[2:].zfill(Binbits)
    TempDeModDataStr.append(NBitsStr)
    for k in range(len(NBitsStr)):
        TempBits.append(int(NBitsStr[k]))
    DeModData = np.array(TempBits)
    DecodedDataBinartStr = ''.join(np.array(TempDeModDataStr, dtype=str))

print('Decoded Data: ', DecodedDataBinartStr)


recoveredStr = CL.text_from_bits(DecodedDataBinartStr)
print(recoveredStr)