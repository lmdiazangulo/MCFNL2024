# %%
import numpy as np
import matplotlib.pyplot as plt

# %% 

t = np.linspace(0, 500e-6, num=int(1e3+1))
s0 = 10e-6
t0 = 10*s0

f = np.exp(- np.power(t - t0,2)/2/s0**2)

plt.plot(t,f , '.-')

# %% 
F =  np.fft.fftshift(np.fft.fft(f))
fq = np.fft.fftshift(np.fft.fftfreq(len(f))/(t[1]-t[0]))

plt.plot(fq, np.abs(F),'.-')
# %%
