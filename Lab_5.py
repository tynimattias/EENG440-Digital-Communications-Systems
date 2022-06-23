from CommunicationLib import DrawSpectrum
import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt


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
plt.plot(t,mt,'r-') 
plt.grid(True)
(markerline, stemlines, baseline) = plt.stem(t,mt) 
plt.setp(baseline, visible=False)
plt.title('Message signal, m(t) and m[n]'); plt.xlabel('t [sec]')
plt.legend(('m(t)','m[n]'))

plt.savefig('Lab_5_Figure_1')
plt.show()


#---------------------------------------------------------------------------



Ac = 2
Kp = 0.1
fc = 1000

Upm = Ac*np.cos(2*np.pi*fc*t + Kp*mt)

plt.figure(2)
plt.subplot(211)
plt.grid()
plt.plot(t, Upm)
plt.subplot(212)
DrawSpectrum(Upm, Fs, 'b')
plt.savefig('Lab_5_Figure_2')
plt.show()


#_____________________________________________________________________________________


N = 257
Fdr = np.array([0.01,0.99])/2
Adr = np.array([1])
B = -sci.remez(N,Fdr,Adr,type='hilbert')

#upm_hat is the hilbert transform of Upm
upm_hat = sci.convolve(B, Upm)

upm_hat = upm_hat[int(np.floor(N/2)):int(np.floor(N/2))+len(mt)]

j = np.complex(0,1)

zt = Upm + (upm_hat*j)

xlt = zt * np.exp(-j*2*np.pi*fc*t)
 

xlt_angle = np.unwrap(np.angle(xlt))


mt_recovered = xlt_angle/Kp




plt.figure(3)
plt.subplot(311)
plt.plot(t, mt, 'ro-')
plt.plot(t, mt_recovered, 'bX-')


plt.subplot(312)
DrawSpectrum(mt, Fs, 'b--')
DrawSpectrum(mt_recovered, Fs, 'r--')



plt.subplot(313)
DrawSpectrum(xlt, Fs, 'k--')
plt.savefig('Lab_5_Figure_3')
plt.show()
