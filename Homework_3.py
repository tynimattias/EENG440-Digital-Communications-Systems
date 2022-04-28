import scipy.signal as sci

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from CommunicationLib import DrawSpectrum

m1t = scipy.io.loadmat('MessageSignal1.mat')  #Then you will see “t” and “m1t” will be loaded.
m2t = scipy.io.loadmat('MessageSignal2.mat')  #Then you will see “t” and “m2t” will be loaded



m1 = m1t['m1t'][0]
m1time = m1t['t'][0]

m2 = m2t['m2t'][0]
m2time = m2t['t'][0]

plt.subplot(211)
plt.plot(m1time, m1)
plt.subplot(212)
plt.plot(m2time, m2)

plt.savefig('HW3_Fig_1')
plt.show()

Ts = m1time[1]-m1time[0]

Fs = 1/Ts

fc_transmitter = 2000 * m1time
ac_transmitter = 1


carrier_transmitter = ac_transmitter*np.cos(2*np.pi*fc_transmitter)

carrier_transmitter_shifted = ac_transmitter*np.cos(2*np.pi*fc_transmitter - (np.pi/2))

e1t = m1 * carrier_transmitter

e2t = m2 * carrier_transmitter_shifted

ut = e1t + e2t


#Reciever

m1_recieved = ut* 2 * carrier_transmitter

m2_recieved = ut * 2 * carrier_transmitter_shifted

#Low-Pass Filter

N=99; CutOffFreq=400
NyqFreq = Fs / 2 # Normalized freq = 1 is corresponding to Nyquist freq, fs/2
NormalizedCutOffFreq = CutOffFreq / NyqFreq
B = sci.firwin(N,NormalizedCutOffFreq)
A = np.array([1])

m1hat = sci.convolve(B,m1_recieved) 
m1hat = m1hat[int(np.floor(N/2)):int(np.floor(N/2))+len(m1time)]

m2hat = sci.convolve(B, m2_recieved)
m2hat = m2hat[int(np.floor(N/2)):int(np.floor(N/2))+len(m2time)]

plt.figure(6)
plt.subplot(211)
DrawSpectrum(m1, Fs, 'r')    
plt.subplot(212)
DrawSpectrum(m2, Fs, 'b')
plt.savefig('HW3_Fig_6')
plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(m1time,ut)
plt.subplot(212)
DrawSpectrum(ut, Fs, 'r--')
plt.savefig('HW3_Fig_2')
plt.show()

#Bandwith needed is about 3000, (eyeballing)

plt.figure(3)
DrawSpectrum(m1, Fs, 'k--')
DrawSpectrum(m1hat, Fs, 'b--')
plt.savefig('HW3_Fig_3')
plt.show()


plt.figure(4)
DrawSpectrum(m2, Fs, 'k--')
DrawSpectrum(m2hat, Fs, 'b--')
plt.savefig('HW3_Fig_4')
plt.show()

plt.figure(5)
plt.subplot(211)
plt.plot(m1time,m1, 'o-')
plt.plot(m1time,m1hat, 'x-')
plt.subplot(212)
plt.plot(m2time,m2, 'o-')
plt.plot(m2time,m2hat, 'x-')
plt.savefig('HW3_Fig_5')
plt.show()

