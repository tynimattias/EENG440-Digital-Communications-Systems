import numpy as np
import scipy.signal as sci
import CommunicationLibLab9 as  CL

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
            try:
                data_to_be_transmitted = self.BitDataToBeTransmitted
            except: AttributeError
            print('Using default data to be transmitted')

        self.data_to_be_transmitted = data_to_be_transmitted
        self.upsampled_BitDataToBeTransmitted = self.Upsampler(self.data_to_be_transmitted, self.sample_value)
        self.h = np.ones(self.sample_value)
        self.h = self.h/np.linalg.norm(self.h)

        filtered_signal = sci.convolve(self.upsampled_BitDataToBeTransmitted, self.h)

        Ts = 0.0002; fs = 1/Ts;
        n = np.arange(0,len(filtered_signal),1)
        self.t = n*Ts
        for n in range(len(self.t)):
            ct = np.cos((2*np.pi*self.fc*self.t[n]) + self.phase)
            filtered_signal[n] = filtered_signal[n]*ct
        self.filtered_signal = filtered_signal
        return self.filtered_signal

    

    def demodulator(self, data_to_be_received = "None"):
        
        if(data_to_be_received=="None"):
            data_to_be_received = self.filtered_signal
        self.data_to_be_received = data_to_be_received
        Ts = 0.0002; fs = 1/Ts;
        n = np.arange(0,len(self.data_to_be_received),1)
        self.t = n*Ts
        for n in range(len(self.t)):
            ct = 2*np.cos((2*np.pi*self.fc*self.t[n]) + self.phase)
            self.data_to_be_received[n] = self.data_to_be_received[n]*ct
        self.received_signal = self.data_to_be_received

        
        self.data_to_be_received = data_to_be_received
        self.filtered_signal = sci.convolve(self.h, self.data_to_be_received)
        self.filtered_signal = self.filtered_signal[len(self.h):len(self.h)+len(self.upsampled_BitDataToBeTransmitted)]
        self.downsampled_signal = self.Downsampler(self.filtered_signal, self.sample_value)
        
        return self.downsampled_signal

    def MinDistDetector(self,MPSKBitLevel):
        # Minimum distiance detector
        N = len(self.downsampled_signal); M = len(MPSKBitLevel)
        d = np.zeros((N,M))
        ds = np.zeros((N,M))
        for k in range(M):
            d[:,k] = (self.downsampled_signal - MPSKBitLevel[k]); ds[:,k] = np.multiply(d[:,k],d[:,k])

        DeModData = np.zeros(N)
        DeModDataStr = np.zeros(N,dtype=np.str)
        
        for n in range(N):
            DeModData[n] = np.argmin(ds[n,:])
            DeModDataStr[n] = np.str(DeModData[n])

        return DeModDataStr, DeModData

    def Downsampler(self,x,M):
        z = np.zeros(int(np.ceil(len(x)/M)))
        i=0
        for nVal in range(len(x)):
            if (nVal % M)==0:
                z[i]=x[nVal]
                i += 1
        return z

    def Upsampler(self,x,M):
        z = np.zeros(M*len(x))
        i=0
        for nVal in range(len(z)):
            if (nVal % M)==0:
                z[nVal]=x[i]
                i += 1
        return z

class MPSK(BPSK):

    def __init__(self, sample_value):
        self.sample_value = sample_value
        
        self.I_Channel = BPSK(self.sample_value, 0)
        self.Q_Channel = BPSK(self.sample_value, np.pi/2)

    def Message_to_Bits(self, message_string):
        self.message = message_string
        BitStringArray = CL.text_to_bits(self.message)
        self.BitDataToBeTransmitted = np.zeros(len(BitStringArray),dtype=np.float64)
        self.Bit0or1Array = np.zeros(len(BitStringArray),dtype=np.int)
        for n in range(len(BitStringArray)):
            if BitStringArray[n] == '1':
                self.BitDataToBeTransmitted[n] = 1
                self.Bit0or1Array[n] = 1
            elif BitStringArray[n] == '0':
                self.BitDataToBeTransmitted[n] = -1
                self.Bit0or1Array[n] = 0
            else:
                print('Error !!! Unknown binary string data!!!')
        return BitStringArray

    def Transmitter(self, message_data = 'None'):
        if message_data == 'None':
            message_data = self.BitDataToBeTransmitted

        







    

