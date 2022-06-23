import CommunicationLibLab9 as mqam
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci
import string
import random

iterations = 100

for i in range(iterations):

    N =300
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = N))

    Str = 'This is EENG 440 Class!!! And EENG 440 is fun!!!'
    BitStringArray = mqam.text_to_bits(res)

    M = 16

    BinBits = int(np.log2(M))

    BinaryDataArray, BinaryNRZ, DecimalDataToBeTransmitted = mqam.BinaryToDecimalArray(BitStringArray,BinBits)


    

    # Dictionary for MPSK encoding
    MPSKGrayEncode = {0:0,1:1,2:3,3:2,4:6,5:7,6:5,7:4, 8:12, 9:13, 10:15, 11:14, 12:10, 13:11, 14:9, 15:8}
    FromOneToTwoBranch = {0:[-3,3], 1:[-1,3], 2:[1,3],3:[3,3],
                        4:[3,1],5:[1,1],
                        6:[-1,1], 7:[-3,1], 
                        8:[-3,-1], 9:[-1,-1], 
                        10:[1,-1], 11:[3,-1], 
                        12:[3,-3], 13:[1,-3], 
                        14:[-1,-3], 15:[-3,-3]}

    IchVal = np.zeros(len(DecimalDataToBeTransmitted))
    QchVal = np.zeros(len(DecimalDataToBeTransmitted))


    for n in range(len(DecimalDataToBeTransmitted)):
        GrayCode = MPSKGrayEncode[DecimalDataToBeTransmitted[n]]
        NBits = FromOneToTwoBranch[GrayCode]
        IchVal[n] = NBits[0]; QchVal[n] = NBits[1]
    
    
    

    # Upsampling
    UpsampledIchVal = mqam.Upsampler(IchVal, 100); UpsampledQchVal = mqam.Upsampler(QchVal, 100)

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


    MeanVal = 0; Variance = pow(10,-3); StdDev = 0.05
    Nse = np.random.normal(MeanVal,StdDev,len(TransmitSignal))
    TransmitSignal = TransmitSignal + Nse

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
    DownsampledIchVal = mqam.Downsampler(FilteredIchVal_Recieved, 100); DownsampledQchVal = mqam.Downsampler(FilteredQchVal_Recieved, 100)

    MPSKBitLevel = [-2, -1, 1, 2]
    IchDeModData, IchDeModStr = mqam.MinDistDetector(DownsampledIchVal, MPSKBitLevel)
    QchDeModData, QchDeModStr = mqam.MinDistDetector(DownsampledQchVal, MPSKBitLevel)


    

    MPSKGrayDecode = {0:0,1:1,3:2,2:3,6:4,7:5,5:6,4:7, 12:8, 13:9, 15:10, 14:11, 10:12, 11:13, 9:14, 8:15}
    FromTwoToOneBranch = np.array([[15,8,7,0],[14,9,6,1],[13,10,5,2],[12,11,4,3]])#------------------------------------------
    #

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


    recoveredStr = mqam.text_from_bits(DecodedDataBinartStr)
    print(recoveredStr)


        

    BitStringArray = mqam.GenerateBinaryBlockData(500)


    NumOfDataInBlock=200; 
    mqam.DrawEyeDiagram(FilteredIchVal_Recieved,NumOfDataInBlock,1)
    plt.title('I-ch eye diagram after receiver filter')
    mqam.DrawEyeDiagram(FilteredQchVal_Recieved,NumOfDataInBlock,2)
    plt.title('Q-ch eye diagram after the receiver filter')
    mqam.DrawScatterPlot(DownsampledIchVal,DownsampledQchVal,4,'*')
    plt.title('Scatter plot for MQAM')

plt.show()
