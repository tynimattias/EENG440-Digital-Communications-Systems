from matplotlib.transforms import TransformNode
import CommunicationLibLab8 as CL
import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt

class BPSK:

    def __init__(self, sample_value, phase=0):
        self.sample_value = sample_value
        self.phase = phase

    def message_to_bits_to_be_transmitted(self, message_str):
        self.message = message_str
        BitStringArray = CL.text_to_bits(self.message)
        self.BitDataToBeTransmitted = np.zeros(len(BitStringArray),dtype=np.float64)
        Bit0or1Array = np.zeros(len(BitStringArray),dtype=np.int)
        for n in range(len(BitStringArray)):
            if BitStringArray[n] == '1':
                self.BitDataToBeTransmitted[n] = 1
                Bit0or1Array[n] = 1
            elif BitStringArray[n] == '0':
                self.BitDataToBeTransmitted[n] = -1
                Bit0or1Array[n] = 0
            else:
                print('Error !!! Unknown binary string data!!!')
        return BitStringArray

    def transmitter(self,fc, data_to_be_transmitted = "None"):
        self.fc = fc
        if(data_to_be_transmitted=="None"):
            data_to_be_transmitted = self.BitDataToBeTransmitted
        self.data_to_be_transmitted = data_to_be_transmitted
        self.upsampled_BitDataToBeTransmitted = CL.Upsampler(self.data_to_be_transmitted, self.sample_value)
        self.h = np.ones(self.sample_value)
        self.h = self.h/np.linalg.norm(self.h)

        filtered_signal = sci.convolve(self.upsampled_BitDataToBeTransmitted, self.h)

        Ts = 0.0002; fs = 1/Ts;
        n = np.arange(0,len(filtered_signal),1)
        self.t = n*Ts # [Hz]
        for n in range(len(self.t)):
            ct = np.cos(2*np.pi*self.fc*self.t[n] + self.phase)
            filtered_signal[n] = filtered_signal[n]*ct

        return filtered_signal
    
    def reciever(self, signal):
        self.recieved_signal = signal

        for n in range(len(self.t)):
            ct = 2*np.cos(2*np.pi*self.fc*self.t[n]+self.phase)
            self.recieved_signal[n] = self.recieved_signal[n]*ct

        recieved_signal_filtered = sci.convolve(self.recieved_signal, self.h)

        recieved_signal_filtered = recieved_signal_filtered[len(self.h):len(self.h)+len(self.upsampled_BitDataToBeTransmitted)]

        downsampled_recieved_signal = CL.Downsampler(recieved_signal_filtered, self.sample_value)

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


        #plt.stem(np.arange(0,len(self.BitDataToBeTransmitted),1),self.BitDataToBeTransmitted, markerfmt="None")
        #plt.stem(np.arange(0,len(DeModData_plot_data),1),DeModData_plot_data, linefmt='c')
        #plt.show()

        return(recieved_signal_filtered, downsampled_recieved_signal)


class MPSK(BPSK):

    def __init__(self, sample_value, phase=0):
        self.sample_value = sample_value
        self.phase = phase
    
#task 0

message = 'This is EENG 440 Class!!! And EENG 440 is fun!!!'
BPSK1 = BPSK(1500)

BitStringArray = BPSK1.message_to_bits_to_be_transmitted(message_str= message)
recieved_message = BPSK1.transmitter(fc = 1500)

BPSK1.reciever(recieved_message)


I = BPSK(100)
Q = BPSK(100)


# Data mapper (i.e., binary bits to decimal numbers)
NumOfBits = 2
BitDataToBeTransmitted,NRZ,DecimalDataToBeTransmitted = CL.BinaryToDecimalArray(BitStringArray,NumOfBits)

# Dictionary for Gray code encoding
QPSKGrayEncode = {0:0,1:1,2:3,3:2}
# Dictionary for QPSK encoding
FromOneToTowBranch = {0:[-1,-1], 1:[-1,1], 2:[1,1], 3:[1,-1]}
print(DecimalDataToBeTransmitted)

DecimalDataToBeTransmitted_Q = np.zeros(len(DecimalDataToBeTransmitted))
DecimalDataToBeTransmitted_I = np.zeros(len(DecimalDataToBeTransmitted))

for i, j in enumerate(DecimalDataToBeTransmitted):
    DecimalDataToBeTransmitted[i] = QPSKGrayEncode[int(j)]
for i, j in enumerate(DecimalDataToBeTransmitted):
    DecimalDataToBeTransmitted_Q[i] = FromOneToTowBranch[int(j)][0]
    DecimalDataToBeTransmitted_I[i] = FromOneToTowBranch[int(j)][1]



print(DecimalDataToBeTransmitted_Q)
print(DecimalDataToBeTransmitted_I)

Q_transmitter_data = I.transmitter(fc = 1500, data_to_be_transmitted=DecimalDataToBeTransmitted_Q)
I_transmitter_data = Q.transmitter(fc = 3000, data_to_be_transmitted= DecimalDataToBeTransmitted_I)

TransmitData = Q_transmitter_data+I_transmitter_data

plt.plot(np.arange(0,len(TransmitData),1),TransmitData)
plt.show()

print(TransmitData)

Recieved_Q_Data, Downsampled_Q = Q.reciever(TransmitData)
Recieved_I_Data, Downsampled_I = I.reciever(TransmitData)


# Minimum distiance detector
QPSKBitLevel = [-1, 1]
IchDeModData,IchDeModDataStr= CL.MinDistDetector(Downsampled_I,QPSKBitLevel)
QchDeModData,QchDeModDataStr= CL.MinDistDetector(Downsampled_Q,QPSKBitLevel)

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
print(BitStringArray)
print(DecodedDataBinaryStr)

print(message)
print(recovered_message)

