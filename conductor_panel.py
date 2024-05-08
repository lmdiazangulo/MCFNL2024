#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from test_fdtd import *
from scipy.constants import speed_of_light

def main():
    sigma_panel = 15.0
    panel_iindex = 50
    panel_findex = 55
    show_animation = True
    
    x = np.linspace(-0.5, 0.5, num = 101) 
    sigma_vector = np.zeros(x.size)
    panel_thickness = panel_findex - panel_iindex
    
    sigma_vector[panel_iindex:panel_findex] = sigma_panel
    fdtd = FDTD1D(x, "mur", sigma_vector = sigma_vector)

    source_length = 0.1
    source = Source.gaussian(10, source_length * 10, 1, source_length/2)
    t_medida = np.arange(0, source_length * 25, step = fdtd.dt)
    fdtd.addSource(source)

    E_incidente = [source.function(t) for t in t_medida]
    E_reflejada = []
    E_transmitida = []

    for _ in t_medida:
        E_reflejada.append(fdtd.getE()[5])
        E_transmitida.append(fdtd.getE()[-10])
        fdtd.step()
        if show_animation:
            plt.plot(fdtd.xE, fdtd.E, '.-')
            plt.plot(fdtd.xH, fdtd.H, '.-')
            plt.vlines(x[[panel_findex, panel_iindex]], -1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.title(fdtd.t)
            plt.grid(which='both')
            plt.pause(0.02)
            plt.cla()

    tSI = t_medida / speed_of_light
    dtSI = fdtd.dt / speed_of_light
    fq = np.fft.fftshift(np.fft.fftfreq(len(t_medida), d = fdtd.dt))
    fqSI = np.fft.fftshift(np.fft.fftfreq(len(tSI), d = dtSI)) 
    Freflejada = np.fft.fftshift(np.fft.fft(E_reflejada)) 
    Fincidente = np.fft.fftshift(np.fft.fft(E_incidente)) 
    Ftransmitida = np.fft.fftshift(np.fft.fft(E_transmitida)) 

    plt.cla()
    plt.plot(fqSI, np.abs(Fincidente),  label = "|E(f)| incidente")
    plt.plot(fqSI, np.abs(Freflejada),  label = "|E(f)| reflejado")
    plt.plot(fqSI, np.abs(Ftransmitida),  label = "|E(f)| transmitido")
    plt.xscale("log")
    plt.legend()
    plt.savefig("freq_domain.png", dpi = 300)
    freq_filter = np.logical_and(np.logical_and(fqSI != 0, fqSI < 20 * speed_of_light), (fqSI > -20 *speed_of_light))
    fqSI = fqSI[freq_filter]
    Freflejada = Freflejada[freq_filter]
    Fincidente = Fincidente[freq_filter]
    Ftransmitida = Ftransmitida[freq_filter]

    panel = Panel(eps_r = 1.0, mu_r = 1.0, sigma = sigma_panel/speed_of_light/MU_0, thickness = panel_thickness * fdtd.dx)
    Rnum = (np.abs(Freflejada)/np.abs(Fincidente))
    Tnum = (np.abs(Ftransmitida)/np.abs(Fincidente))

    w = 2 * np.pi * fqSI
    R = np.abs([panel.getReflectionCoefficient(w) for w in w])
    T = np.abs([panel.getTransmissionCoefficient(w) for w in w])
    plt.cla()
    plt.plot(fqSI, Rnum, '.', label = 'R numérico')
    plt.plot(fqSI, Tnum, '.', label = 'T numérico')
    plt.plot(fqSI, R, label = 'R')
    plt.plot(fqSI, T, label = 'T')
    plt.xscale("log")
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig("coeffs.png", dpi = 300)
    plt.cla()
    plt.plot(tSI, E_incidente,  label = "E incidente")
    plt.plot(tSI, E_reflejada,  label = "E reflejado")
    plt.plot(tSI, E_transmitida,  label = "E transmitido")
    plt.legend()
    plt.savefig("time_domain.png", dpi = 300)

    
if __name__ == '__main__':
    main()
