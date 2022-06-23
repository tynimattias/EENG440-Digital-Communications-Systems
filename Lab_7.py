import CommunicationLib as CL
import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt
from scipy.io import loadmat

sample = 100

# Make a message to be transmitted.
Str = 'EENG 440 test string Lab 7 '
BitStringArray = CL.text_to_bits(Str)

# Binary mapper (i.e, from [0,1] to [-1 ])
BitDataToBeTransmitted = np.zeros(len(BitStringArray),dtype=np.float64)
for n in range(len(BitStringArray)):
    if BitStringArray[n] == '1':
        BitDataToBeTransmitted[n] = 1
    elif BitStringArray[n] == '0': 
        BitDataToBeTransmitted[n] = -1
    else:
        print('Error !!! Unknown binary string data!!!')

binary_message_data = CL.Upsampler(BitDataToBeTransmitted, sample)

t = np.arange(-6,6,0.01)

transmitting_filter = np.sinc(t)
transmitting_filter/np.linalg.norm(transmitting_filter)

st = sci.convolve(binary_message_data, transmitting_filter)




st_adjusted = st[len(transmitting_filter):len(transmitting_filter)+len(binary_message_data)]
st_adjusted_index = np.arange(0,len(st_adjusted),1)

#_______________________________________

recieving_filter = transmitting_filter

yt = sci.convolve(st, recieving_filter)
yt = yt[len(transmitting_filter):len(transmitting_filter)+len(binary_message_data)]

downsampled_yt = CL.Downsampler(yt, sample)

print(f'length of downsampled is {len(downsampled_yt)}')
print(f'transmitteed = {len(BitDataToBeTransmitted)}')


r = np.array([-1,1])

#Detector

detected =[]


 

for i in range(len(downsampled_yt)):
    
    j1 = (downsampled_yt[i]-1)**2
    j2 = (downsampled_yt[i]+1)**2
    

    if(j2>j1):
        detected.append(str(1))
       # print('a')
    elif(j1>j2):
        detected.append(str(0))
       # print('b')
    else:
       # print('c')
        detected.append(str(np.random.randint(0,2)))
   # print(f'j1 was {j1}')
   # print(f'j2 was {j2}')

detected_joined = ''.join(detected)
print(detected_joined)
print(BitStringArray)

text = CL.text_from_bits(detected_joined)
text_og = CL.text_from_bits(BitStringArray)
print(text)
print(text_og)

for i in range(len(detected)):
    if(detected[i]=='0'):
        detected[i]=-1
    else:
        detected[i]=1

index = np.arange(0,len(detected),1)

plt.subplot(311)
plt.stem(index, detected)
plt.stem(index, BitDataToBeTransmitted, 'g--')

plt.subplot(312)
plt.plot(t, transmitting_filter)

plt.subplot(313)
plt.plot(st_adjusted_index, st_adjusted)

plt.show()


#____________________________________________________________________



# Transmit filter
t = np.arange(-6,6,0.01)
# To see an ISI, use the following "h"
Temp = loadmat('EENG_440\ArbitraryFilter.mat')
Temp = Temp['h']
h = Temp[0] #Temp[0][:-1]
h = h/np.linalg.norm(h)


st_isi = sci.convolve(binary_message_data, h)

st_isi_adjusted = st[len(h):len(h)+len(binary_message_data)]
st_isi_index = np.arange(0,len(st_isi_adjusted),1)


yt = sci.convolve(st, h)
yt = yt[len(h):len(h)+len(binary_message_data)]

downsampled_yt = CL.Downsampler(yt, sample)

print(f'length of downsampled is {len(downsampled_yt)}')
print(f'transmitteed = {len(BitDataToBeTransmitted)}')


r = np.array([-1,1])

#Detector

detected =[]


 

for i in range(len(downsampled_yt)):
    
    j1 = (downsampled_yt[i]-1)**2
    j2 = (downsampled_yt[i]+1)**2
    

    if(j2>j1):
        detected.append(str(1))
       # print('a')
    elif(j1>j2):
        detected.append(str(0))
       # print('b')
    else:
       # print('c')
        detected.append(str(np.random.randint(0,2)))
   # print(f'j1 was {j1}')
   # print(f'j2 was {j2}')

detected_joined = ''.join(detected)
print(detected_joined)
print(BitStringArray)

text = CL.text_from_bits(detected_joined)
text_og = CL.text_from_bits(BitStringArray)
print(text)
print(text_og)

for i in range(len(detected)):
    if(detected[i]=='0'):
        detected[i]=-1
    else:
        detected[i]=1

index = np.arange(0,len(detected),1)

t = np.append(t,6)

plt.subplot(411)
plt.stem(index, detected)
plt.stem(index, BitDataToBeTransmitted, 'g--')

plt.subplot(412)
plt.plot(t, h)

plt.subplot(413)
plt.plot(st_isi_index, st_isi_adjusted)

plt.subplot(414)
plt.plot(np.arange(0,len(yt),1),yt )


plt.show()


#________________________________________________________________________________________________________________________________________



# Make a message to be transmitted.
Str = 'EENG 440 test string Lab 7'
BitStringArray = CL.text_to_bits(Str)

# Binary mapper (i.e, from [0,1] to [-1 ])
BitDataToBeTransmitted = np.zeros(len(BitStringArray),dtype=np.float64)
for n in range(len(BitStringArray)):
    if BitStringArray[n] == '1':
        BitDataToBeTransmitted[n] = 1
    elif BitStringArray[n] == '0':
        BitDataToBeTransmitted[n] = -1
    else:
        print('Error !!! Unknown binary string data!!!')

binary_message_data = CL.Upsampler(BitDataToBeTransmitted, sample)

t = np.arange(-6,6,0.01)

transmitting_filter = np.sinc(t)
transmitting_filter = transmitting_filter/np.linalg.norm(transmitting_filter)

st = sci.convolve(binary_message_data, transmitting_filter)



st_adjusted = st[len(transmitting_filter):len(transmitting_filter)+len(binary_message_data)]
st_adjusted_index = np.arange(0,len(st_adjusted),1)


MeanVal = 0; StdDev = 0.05 #Originally 0.5
Nse = np.random.normal(MeanVal,StdDev,len(st))
st = st + Nse

print(Nse)


#_______________________________________


yt = sci.convolve(st, transmitting_filter)
yt = yt[len(transmitting_filter):len(transmitting_filter)+len(binary_message_data)]

downsampled_yt = CL.Downsampler(yt, sample)

print(f'length of downsampled is {len(downsampled_yt)}')
print(f'transmitteed = {len(BitDataToBeTransmitted)}')


r = np.array([-1,1])

#Detector

detected =[]


 

for i in range(len(downsampled_yt)):
    
    j1 = (downsampled_yt[i]-1)**2
    j2 = (downsampled_yt[i]+1)**2
    

    if(j2>j1):
        detected.append(str(1))
       # print('a')
    elif(j1>j2):
        detected.append(str(0))
       # print('b')
    else:
       # print('c')
        detected.append(str(np.random.randint(0,2)))
   # print(f'j1 was {j1}')
   # print(f'j2 was {j2}')

detected_joined = ''.join(detected)
print(detected_joined)
print(BitStringArray)

text = CL.text_from_bits(detected_joined)
text_og = CL.text_from_bits(BitStringArray)
print(text)
print(text_og)

for i in range(len(detected)):
    if(detected[i]=='0'):
        detected[i]=-1
    else:
        detected[i]=1

index = np.arange(0,len(detected),1)

plt.subplot(311)
plt.stem(index, detected)
plt.stem(index, BitDataToBeTransmitted, 'g--')

plt.subplot(312)
plt.plot(t, transmitting_filter)

plt.subplot(313)
plt.plot(st_adjusted_index, st_adjusted)

plt.show()



NumOfDataInBlock=200;
CL.DrawEyeDiagram(st,NumOfDataInBlock,4)
plt.title('Eye diagram after transmit filter')
CL.DrawEyeDiagram(yt,NumOfDataInBlock,5)
plt.title('Eye diagram after the receiver filter')
Qch = np.zeros(len(downsampled_yt))
CL.DrawScatterPlot(downsampled_yt,Qch,6,'*')
plt.title('Scatter plot for PAM')
plt.show()