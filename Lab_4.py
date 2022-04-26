from CommunicationLib import DrawSpectrum
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sci


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

plt.savefig('Figure 1')
plt.show()


N = 257
Fdr = np.array([0.01,0.99])/2
Adr = np.array([1])
B = -sci.remez(N,Fdr,Adr,type='hilbert')

#B is the filter for the Hilbert Transform


Bd = np.zeros(N) 
Bd[int(np.floor(N/2))]=1
mtDelayed = sci.convolve(Bd,mt)

xhat = sci.convolve(B, mt)

#xhat is our message signal with the Hilbert Transform filter appied, resulting in the Hilbert Transform of the message signal

#M
xhat = xhat[int(np.floor(N/2)):int(np.floor(N/2))+len(mt)]

plt.figure(2)
plt.plot(t,xhat)
plt.grid(True)
plt.savefig('Figure 2')
plt.show()

fc = 1000

ct = np.cos(2*np.pi*fc*t)

et = mt * ct

ct_phaseshift = np.cos(2*np.pi*fc*22*t - (np.pi/2))

ethat = xhat * ct_phaseshift


ut = et - ethat

plt.figure(3)
DrawSpectrum(mt, Fs, 'b--')
DrawSpectrum(ut, Fs, 'r--')
plt.savefig('Figure_3')
plt.show()

rt = ut

et_recovered = ut*2*ct

N=99; CutOffFreq=400
NyqFreq = Fs / 2 # Normalized freq = 1 is corresponding to Nyquist freq, fs/2
NormalizedCutOffFreq = CutOffFreq / NyqFreq
B = sci.firwin(N,NormalizedCutOffFreq)
A = np.array([1])

yt = sci.convolve(B,et_recovered) 
yt = yt[int(np.floor(N/2)):int(np.floor(N/2))+len(t)]

plt.figure(4)
plt.subplot(3,1,1)
DrawSpectrum(B*200, Fs, 'b')
DrawSpectrum(et_recovered, Fs, 'r--')

plt.subplot(3,1,2)
DrawSpectrum(yt, Fs, 'r--')
DrawSpectrum(mt, Fs, 'b--')

plt.subplot(3,1,3)
plt.plot(t, mt)
plt.plot(t, yt)
plt.grid(True)

plt.savefig('Figure_4')
plt.show()