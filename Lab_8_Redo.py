from Lab_8 import BPSK
import CommunicationLibLab8 as CL
import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt


sampler_value = 100
#--------------Part 0-------------------------------

Str = 'This is EENG 440 Class!!! And EENG 440 is fun!!!'
BitStringArray = CL.text_to_bits(Str)

#-------------Part 1----------------------------------Transmitter BPSK


BitDataToBeTransmitted = np.zeros(len(BitStringArray),dtype=np.float64)
Bit0or1Array = np.zeros(len(BitStringArray),dtype=np.int)
for n in range(len(BitStringArray)):
    if BitStringArray[n] == '1':
        BitDataToBeTransmitted[n] = 1
        Bit0or1Array[n] = 1
    elif BitStringArray[n] == '0':
        BitDataToBeTransmitted[n] = -1
        Bit0or1Array[n] = 0
    else:
        print('Error !!! Unknown binary string data!!!')


upsampled_BitDataToBeTransmitted = CL.Upsampler(BitDataToBeTransmitted, sampler_value)

h = np.ones(sampler_value)
h = h/np.linalg.norm(h)

filtered_signal = sci.convolve(upsampled_BitDataToBeTransmitted, h)

Ts = 0.0002; fs = 1/Ts;
n = np.arange(0,len(filtered_signal),1)
t = n*Ts
fc=1600 # [Hz]
for n in range(len(t)):
    ct = np.cos(2*np.pi*fc*t[n])
    filtered_signal[n] = filtered_signal[n]*ct



#-----------------------Part 2--------------------------------------Reciever BPSK

recieved_signal = filtered_signal

for n in range(len(t)):
    ct = np.cos(2*np.pi*fc*t[n])
    recieved_signal[n] = recieved_signal[n]*ct

recieved_signal_filtered = sci.convolve(recieved_signal, h)

recieved_signal_filtered = recieved_signal_filtered[len(h):len(h)+len(upsampled_BitDataToBeTransmitted)]

downsampled_recieved_signal = CL.Downsampler(recieved_signal_filtered, sampler_value)

# Minimum distiance detector
BPSKBitLevel = np.array([-1, 1]);
[BitLevelIndx,BitLevelIndxStr] = CL.MinDistDetector(downsampled_recieved_signal,BPSKBitLevel);


# ------ Dictionary for BPSK decoding -------
ValueToBeChanged = np.array([0,1]);
# Decode data using the recovered bits
TempBits = []; TempDeModDataStr = []
for n in range(len(BitLevelIndx)):
    NBits = ValueToBeChanged[int(BitLevelIndx[n])]
    TempDeModDataStr.append(str(NBits))
    TempBits.append(int(NBits))
DeModData = np.array(TempBits)
DeModDataStr = ''.join(np.array(TempDeModDataStr,dtype=np.str))

DeModData_plot_data = np.zeros(len(DeModData))
for h in range(len(DeModData)):
    if(DeModData[h]==1):
        DeModData_plot_data[h]=1
    else:
        DeModData_plot_data[h]=-1



# Recover the received message from the decoded data bits
RecoveredStr = CL.text_from_bits(''.join(DeModDataStr))
print(RecoveredStr)


plt.stem(np.arange(0,len(BitDataToBeTransmitted),1),BitDataToBeTransmitted, markerfmt="None")
plt.stem(np.arange(0,len(DeModData_plot_data),1),DeModData_plot_data, linefmt='c')
plt.show()




#---------------------Part 3-------------------------QPSK Transmitter

# Data mapper (i.e., binary bits to decimal numbers)
NumOfBits = 2
BitDataToBeTransmitted,NRZ,DecimalDataToBeTransmitted = CL.BinaryToDecimalArray(BitStringArray,NumOfBits)

# Dictionary for Gray code encoding
QPSKGrayEncode = {0:0,1:1,2:3,3:2}
# Dictionary for QPSK encoding
FromOneToTowBranch = {0:[-1,-1], 1:[-1,1], 2:[1,1], 3:[1,-1]}

codded_DecimalData = np.zeros(len(DecimalDataToBeTransmitted))

for i,j in enumerate(DecimalDataToBeTransmitted):
    codded_DecimalData[i] = QPSKGrayEncode[j]


I_Channel_data = np.zeros(len(codded_DecimalData))
Q_Channel_data = np.zeros(len(codded_DecimalData))

for i, j in enumerate(codded_DecimalData):
    I_Channel_data[i] = FromOneToTowBranch[j][0]
    Q_Channel_data[i] = FromOneToTowBranch[j][1]


upsampled_I_Channel = CL.Upsampler(I_Channel_data, sampler_value)
upsampled_Q_Channel = CL.Upsampler(Q_Channel_data, sampler_value)

#Transmit filter is h from part 1

h = np.ones(sampler_value)
h = h/np.linalg.norm(h)

filtered_I_Channel = sci.convolve(h, upsampled_I_Channel)
filtered_Q_Channel = sci.convolve(h, upsampled_Q_Channel)


Ts = 0.0002; fs = 1/Ts;
n = np.arange(0,len(filtered_I_Channel),1)
t = n*Ts
fc=1600 # [Hz]
for n in range(len(t)):
    ct = np.cos(2*np.pi*fc*t[n])
    filtered_I_Channel[n] = filtered_I_Channel[n]*ct
    ct = np.sin(2*np.pi*fc*t[n])
    filtered_Q_Channel[n] = filtered_Q_Channel[n]*ct

TransmitSignal = filtered_I_Channel + filtered_Q_Channel

#---------Part 4-----------------------------
Recieved_I_Channel = np.zeros(len(filtered_I_Channel))
Recieved_Q_Channel = np.zeros(len (filtered_Q_Channel))


for n in range(len(t)):
    ct = np.cos(2*np.pi*fc*t[n])
    Recieved_I_Channel[n] = TransmitSignal[n]*ct
    ct = np.sin(2*np.pi*fc*t[n])
    Recieved_Q_Channel[n] = TransmitSignal[n]*ct

filtered_recieved_Q_Channel = sci.convolve(Recieved_Q_Channel, h)
filtered_recieved_I_Channel = sci.convolve(Recieved_I_Channel, h)

filtered_recieved_Q_Channel = filtered_recieved_Q_Channel[len(h):len(h)+len(upsampled_Q_Channel)]
filtered_recieved_I_Channel = filtered_recieved_I_Channel[len(h):len(h)+len(upsampled_I_Channel)]

downsampled_recieved_Q_Channel = CL.Downsampler(filtered_recieved_Q_Channel, sampler_value)
downsampled_recieved_I_Channel = CL.Downsampler(filtered_recieved_I_Channel, sampler_value)

# Minimum distiance detector
QPSKBitLevel = [-1, 1]
IchDeModData,IchDeModDataStr= CL.MinDistDetector(downsampled_recieved_I_Channel,QPSKBitLevel)
QchDeModData,QchDeModDataStr= CL.MinDistDetector(downsampled_recieved_Q_Channel,QPSKBitLevel)

IchDeModData_plot = np.zeros(len(IchDeModData))

for i in range(len(IchDeModData_plot)):
    if(IchDeModData[i]==0):
        IchDeModData_plot[i]=-1
    else:
        IchDeModData_plot[i]=1

QchDeModData_plot = np.zeros(len(QchDeModData))

for i in range(len(IchDeModData_plot)):
    if(QchDeModData[i]==0):
        QchDeModData_plot[i]=-1
    else:
        QchDeModData_plot[i]=1


# Dictionary for QPSK decoding
FromTwoToOneBranch = np.array([[0,1],[3,2]])
QPSKGrayDecode= {0:0,1:1,2:3,3:2}

# Decode data using the recovered Ich and Qch
TempBits = []; TempDeModDataStr = []
for n in range(len(IchDeModData)):
    TempDec = FromTwoToOneBranch[int(IchDeModData[n]),int(QchDeModData[n])]
    DecodedGrayDecimal = QPSKGrayDecode[TempDec]
    NBitsStr = bin(DecodedGrayDecimal)[2:].zfill(NumOfBits)
    TempDeModDataStr.append(NBitsStr)
    for k in range(len(NBitsStr)):
        TempBits.append(int(NBitsStr[k]))
DeModData = np.array(TempBits)
DecodedDataBinaryStr = ''.join(np.array(TempDeModDataStr,dtype=np.str))

recovered_message = CL.text_from_bits(DecodedDataBinaryStr)



print('Original String was: '+Str)
print('Recovered Message was: ' +recovered_message)

plt.subplot(211)
plt.stem(np.arange(0,len(I_Channel_data),1),I_Channel_data, 'k--')
plt.stem(np.arange(0,len(IchDeModData_plot),1),IchDeModData_plot)

plt.subplot(212)
plt.stem(np.arange(0,len(Q_Channel_data),1),Q_Channel_data,'k--')
plt.stem(np.arange(0,len(QchDeModData_plot),1),QchDeModData_plot)

plt.show()

#------------------Part 5-----------------------------------------

BinaryLen = 500

BitStringArray = CL.GenerateBinaryBlockData(BinaryLen)

NumOfBits = 2
BitDataToBeTransmitted,NRZ,DecimalDataToBeTransmitted = CL.BinaryToDecimalArray(BitStringArray,NumOfBits)

print(DecimalDataToBeTransmitted)

#-------

codded_DecimalData = np.zeros(len(DecimalDataToBeTransmitted))

for i,j in enumerate(DecimalDataToBeTransmitted):
    codded_DecimalData[i] = QPSKGrayEncode[j]


I_Channel_data = np.zeros(len(codded_DecimalData))
Q_Channel_data = np.zeros(len(codded_DecimalData))

for i, j in enumerate(codded_DecimalData):
    I_Channel_data[i] = int(FromOneToTowBranch[j][0])
    Q_Channel_data[i] = int(FromOneToTowBranch[j][1])


I = BPSK(100,0)
Q = BPSK(100,-np.pi/2)

I_data = I.transmitter(1500, I_Channel_data)
Q_data = Q.transmitter(1500, Q_Channel_data)


#-----

TransmittingSig = I_data+Q_data


MeanVal = 0; StdDev = 0.1
Nse = np.random.normal(MeanVal,StdDev,len(TransmittingSig))
print(Nse)

TransmittingSig = TransmittingSig + Nse
#----------------------------------------------------------------------------------------

a,Downsample_I = I.reciever(TransmittingSig)
b,Downsample_Q = Q.reciever(TransmittingSig)




NumOfDataInBlock=200;

CL.DrawEyeDiagram(a,NumOfDataInBlock,1)
plt.savefig('EyeDiagram_I.png')
plt.title('I-ch eye diagram after receiver filter')
CL.DrawEyeDiagram(b,NumOfDataInBlock,2)
plt.title('Q-ch eye diagram after the receiver filter')
CL.DrawScatterPlot(Downsample_I,Downsample_Q,3,'*')
plt.title('Scatter plot for QPSK')
plt.show()
