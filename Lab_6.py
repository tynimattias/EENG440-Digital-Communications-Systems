from CommunicationLib import DrawSpectrum
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sci
import scipy.integrate as integrate


length = 200

f = [30, 70, 150, 190]
Ts = 0.0002
Fs = 1/Ts

n = np.arange(0,length,1)
t = Ts * n

mt = np.zeros(length)
msgtemp = np.zeros(length)


for i in range(len(f)):
  for k in range(len(msgtemp)):
    msgtemp[k]=3*np.sin(2*np.pi*f[i]*t[k])
  mt += msgtemp
  msgtemp = np.zeros(len(msgtemp))
  


Fig1 = plt.figure(1)
plt.subplot(311) 
plt.plot(t,mt,'r-') 
plt.grid(True)
(markerline, stemlines, baseline) = plt.stem(t,mt) 
plt.setp(baseline, visible=False)
plt.title('Message signal, m(t) and m[n]'); plt.xlabel('t [sec]')
plt.legend(('m(t)','m[n]'))
plt.tight_layout()
plt.savefig('Figure 1')
plt.show()
#_______________________________________________________________________Part 2


fc = 1500
Ac = 2
K_w = 0.01 #k_w = 2pik_f
print(len(mt))




#np.cumsum to calculate the running integral of mt
m_t_integral = np.cumsum(mt)

phi_t = K_w * m_t_integral

U_FM = Ac*np.cos((2*np.pi*fc*t)+(phi_t))

plt.figure(2)
plt.subplot(211)
plt.plot(t, U_FM)
plt.tight_layout()

plt.subplot(212)
DrawSpectrum(U_FM, Fs, 'b')
plt.tight_layout()

plt.show()

#________________________________________________________________________________Part 3


N = 257
Fdr = np.array([0.01,0.99])/2
Adr = np.array([1])
B = -sci.remez(N,Fdr,Adr,type='hilbert')

#upm_hat is the hilbert transform of Upm
U_FM_hat = sci.convolve(B, U_FM)
U_FM_hat = U_FM_hat[int(np.floor(N/2)):int(np.floor(N/2))+len(mt)]



j = np.complex(0,1)

zt = U_FM + (U_FM_hat*j)

xlt = zt * np.exp(-j*2*np.pi*fc*t)
 

xlt_angle = np.unwrap(np.angle(xlt))

recovered = np.diff(xlt_angle, prepend=[0])/K_w



plt.figure(3)
plt.subplot(311)
plt.plot(t, mt)
plt.plot(t, recovered)
plt.tight_layout()

plt.subplot(312)
DrawSpectrum(recovered, Fs, 'r--')
DrawSpectrum(mt,Fs, 'b--')
plt.tight_layout()


plt.subplot(313)
DrawSpectrum(xlt, Fs, 'g')
plt.tight_layout()
plt.show()

#_________________________________________________________________________________________Part 4

N=99; CutOffFreq=350
NyqFreq = Fs / 2 # Normalized freq = 1 is corresponding to Nyquist freq, fs/2
NormalizedCutOffFreq = CutOffFreq / NyqFreq
B = sci.firwin(N,NormalizedCutOffFreq)
A = np.array([1])

recovered_message = sci.convolve(B, recovered)
recovered_message = recovered_message[int(np.floor(N/2)):int(np.floor(N/2))+len(mt)]

plt.figure(4)
plt.subplot(211)
plt.plot(t, mt)
plt.plot(t, recovered_message)

plt.subplot(212)
DrawSpectrum(recovered_message, Fs, 'r--')
DrawSpectrum(B*180, Fs, 'k')
DrawSpectrum(mt, Fs, 'b--')

plt.tight_layout()
plt.show()