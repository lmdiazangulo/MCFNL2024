import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light


EPSILON_0 = 8.85e-12
MU_0 = np.pi * 4 * 1e-7
ETA_0 = np.sqrt(MU_0/EPSILON_0)
class Panel():
    def __init__(self, eps_r, mu_r, sigma, thickness):
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma = sigma
        self.thickness = thickness
        
    def phi(self, w):
        mu = self.mu_r * MU_0
        epsC = self.eps_r * EPSILON_0 - 1j*self.sigma/w
        # print(epsC)
        gamma = 1j * w *np.sqrt(mu * epsC)
        # print(self.sigma/w)
        gd = gamma*self.thickness
        eta = np.sqrt(mu/epsC)
        return np.array(
                [
                [np.cosh(gd),     eta * np.sinh(gd)],
                [1/eta*np.sinh(gd),     np.cosh(gd)]
                ]
            )
        

    def denominator(self, w):
        return self.phi(w)[0,0] * ETA_0 + self.phi(w)[0,1] + self.phi(w)[1,0]*ETA_0**2 + self.phi(w)[1,1]*ETA_0
    
    def getTransmissionCoefficient(self, w):
        return 2*ETA_0/self.denominator(w)
        
    def getReflectionCoefficient(self, w):
        return (self.phi(w)[0,0] * ETA_0 + self.phi(w)[0,1] - self.phi(w)[1,0]*ETA_0**2 - self.phi(w)[1,1]*ETA_0)/(
                self.denominator(w)
        )

class FDTD1D():
    def __init__(self, xE, boundary, relative_epsilon_vector=None, sigma_vector = None, doplot = False):
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0
        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)
        self.dx = xE[1] - xE[0]
        self.dt = 1.0 * self.dx
        self.doplot = doplot
        
        self.sources = []
        self.t = 0.0
        if sigma_vector is None:
            self.sigma_vector = np.zeros(self.xE.shape)
        else:
            self.sigma_vector = sigma_vector
        if relative_epsilon_vector is None:
            self.epsilon_r = np.ones(self.xE.shape)
        else:
            self.epsilon_r = relative_epsilon_vector
        self.boundary = boundary

    def addSource(self, source):
        self.sources.append(source)
     
    def setE(self, fieldE):
        self.E[:] = fieldE[:]

    def setH(self, fieldH):
        self.H[:] = fieldH[:]

    def getE(self):
        fieldE = np.zeros(self.E.shape)
        fieldE = self.E[:]
        return fieldE
    
    def getH(self):
        fieldH = np.zeros(self.H.shape)
        fieldH = self.H[:]
        return fieldH

    def step(self):
        E = self.E
        H = self.H
        c0 = self.dt/self.dx
        c_eps = np.ones(self.epsilon_r.size)

        c_eps[:] = c0 / self.epsilon_r[:]
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        H += - c0 *(E[1:] - E[:-1])
        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)

        epsilon = self.epsilon_r 
        E_factor = (epsilon[1:-1] - self.sigma_vector[1:-1]*self.dt/2)\
            /(epsilon[1:-1] + self.sigma_vector[1:-1]*self.dt/2)
        H_factor = self.dt/(epsilon[1:-1] + self.sigma_vector[1:-1]*self.dt/2) * (1/self.dx)
        E[1:-1] =  E_factor * E[1:-1] - H_factor * (H[1:] - H[:-1])
        for source in self.sources:
            E[source.location] += source.function(self.t + self.dt - self.dx/2)
        self.t += self.dt

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c0 / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c0 / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        elif self.boundary == "mur":
            cte = (c0-1.0)/(c0 + 1.0)
            # Left 
            E[0] = E_aux_izq + cte*( E[1] - E[0])
            # Right
            E[-1] = E_aux_dch + cte*( E[-2] - E[-1] )
        else:
            raise ValueError("Boundary not defined")

    def run_until(self, finalTime):
        while (self.t < finalTime):
            if self.doplot:    
                plt.plot(self.xE, self.E, '.-')
                plt.plot(self.xH, self.H, '.-')
                plt.ylim(-1.1, 1.1)
                plt.title(self.t)
                plt.grid(which='both')
                plt.pause(0.02)
                plt.cla()
            self.step()

class Source():
    def __init__(self, location, function):
        self.location = location
        self.function = function
    def gaussian(location, center, amplitude, spread):
        def function(t):
            return np.exp( - (((t-center)/spread)**2)/2.0) * amplitude
        return Source(location, function)
    def square(location, tini, tfin, amplitude):
        def function(t):
            if t > tini and t < tfin:
                return amplitude 
            else:
                return 0
        return Source(location, function)
          


def test_pec():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "pec")

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(R[0,1], 1.0)

def test_pmc():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "pmc")

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R[0,1], 1.0)

def test_period():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "period")

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(1.0)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)


def test_pec_dielectric():
    x = np.linspace(-0.5, 0.5, num=101)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(x.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(x, "pec", epsilon_vector)

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(time)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(np.abs(R[0,1]), 1.0)

def test_period_dielectric():
    x = np.linspace(-0.5, 0.5, num=101)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(x.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)
    
    fdtd = FDTD1D(x, "period", epsilon_vector)

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(time)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)

def test_mur():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "mur")

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.1)

    assert np.allclose(fdtd.getE(), np.zeros_like(fdtd.getE()), atol = 1.e-2)

def test_error():
    error = np.zeros(5)
    deltax = np.zeros(5)
    for i in range(5):
        num = 10**(i+1) +1
        x = np.linspace(-0.5, 0.5, num)
        deltax[i] = 1/num
        fdtd = FDTD1D(x, "pec")
        spread = 0.1
        initialE = np.exp( - ((x-0.1)/spread)**2/2)
        
        fdtd.setE(initialE)
        fdtd.step()
        fdtd.step()
        N = len(initialE)
        error[i] = np.sqrt(np.sum((fdtd.getE() - initialE)**2)) / N
        
    # plt.plot(deltax, error)
    # plt.loglog()
    # plt.grid(which='both')
    # plt.show()
    
    # np.polyfit(np.log10(error), np.log10(deltax), 1)
    
    slope = (np.log10(error[-1]) - np.log10(error[0])) / \
        (np.log10(deltax[-1]) - np.log10(deltax[0]) )


    assert np.isclose( slope , 2, rtol=1.e-1)
                                
def test_illumination():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "pec")
    finalTime = 1.0

    fdtd.addSource(Source.gaussian(20, 0.5, 0.5, 0.1))
    fdtd.addSource(Source.gaussian(70, 1.0, -0.5, 0.1))

    while (fdtd.t <= finalTime):
        fdtd.step()
        assert np.isclose(fdtd.getE()[5], 0.0, atol=1e-5)
    assert np.allclose(fdtd.getE()[:20], 0.0, atol = 1e-5)
    assert np.allclose(fdtd.getE()[71:], 0.0, atol = 1e-5)
    

def test_pmc():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "pmc")

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.0)
    R = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R[0,1], 1.0)

def test_conductor_panel():
    x = np.linspace(-0.5, 0.5, num = 101) 
    sigma_vector = np.zeros(x.size)
    sigma_panel = 15.0
    sigma_vector[50:55] = sigma_panel
    fdtd = FDTD1D(x, "mur", sigma_vector = sigma_vector)

    t_medida = np.arange(0, 6, step = fdtd.dt)

    source = Source.gaussian(10, 0.5, 1, 0.05)
    fdtd.addSource(source)

    E_incidente = [source.function(t) for t in t_medida]
    E_reflejada = []
    E_transmitida = []

    for _ in t_medida:
        E_reflejada.append(fdtd.getE()[5])
        E_transmitida.append(fdtd.getE()[-10])
        fdtd.step()
        # plt.plot(fdtd.xE, fdtd.E, '.-')
        # # plt.plot(self.xH, self.H, '.-')
        # plt.ylim(-1.1, 1.1)
        # plt.title(fdtd.t)
        # plt.grid(which='both')
        # plt.pause(0.02)
        # plt.cla()

    tSI = t_medida / speed_of_light
    dtSI = fdtd.dt / speed_of_light
    fq = np.fft.fftshift(np.fft.fftfreq(len(t_medida), d = fdtd.dt))
    fqSI = np.fft.fftshift(np.fft.fftfreq(len(tSI), d = dtSI)) 
    Freflejada = np.fft.fftshift(np.fft.fft(E_reflejada)) 
    Fincidente = np.fft.fftshift(np.fft.fft(E_incidente)) 
    Ftransmitida = np.fft.fftshift(np.fft.fft(E_transmitida)) 

    freq_filter = np.logical_and(np.logical_and(fqSI != 0, fqSI < 20 * speed_of_light), (fqSI > -20 *speed_of_light))
    fqSI = fqSI[freq_filter]
    Freflejada = Freflejada[freq_filter]
    Fincidente = Fincidente[freq_filter]
    Ftransmitida = Ftransmitida[freq_filter]
    panel = Panel(eps_r = 1.0, mu_r = 1.0, sigma = sigma_panel/speed_of_light/MU_0, thickness = 5 * fdtd.dx)
    Rnum = (np.abs(Freflejada)/np.abs(Fincidente))
    Tnum = (np.abs(Ftransmitida)/np.abs(Fincidente))

    w = 2 * np.pi * fqSI
    R = np.abs([panel.getReflectionCoefficient(w) for w in w])
    T = np.abs([panel.getTransmissionCoefficient(w) for w in w])
    # plt.plot(fqSI, Rnum, '.', label = 'R numérico')
    # plt.plot(fqSI, Tnum, '.', label = 'T numérico')
    # plt.plot(fqSI, R, label = 'R')
    # plt.plot(fqSI, T, label = 'T')
    assert np.allclose(Rnum, R, atol=2e-2)
    assert np.allclose(Tnum, T, atol=2e-2)
    # plt.legend()
    # plt.ylim(0, 1.1)
    # plt.show()

def test_conductivity_absorption():
    x = np.linspace(-0.5, 0.5, num = 101) 
    sigma_vector = np.zeros(x.size)
    sigma_vector[:] = 1.0
    fdtd = FDTD1D(x, "pec", sigma_vector = sigma_vector)

    fdtd.addSource(Source.gaussian(10, 0.5, 0.5, 0.05))
    fdtd.run_until(1.0)
    energy0 = np.sum(fdtd.getE()**2) + np.sum(fdtd.getH()**2)
    fdtd.run_until(5.0)
    energy1 = np.sum(fdtd.getE()**2) + np.sum(fdtd.getH()**2)
    assert energy1 < energy0


def test_period():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x, "period")

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(1.0)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)


def test_pec_dielectric():
    x = np.linspace(-0.5, 0.5, num=101)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(x.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(x, "pec", epsilon_vector)

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(time)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(np.abs(R[0,1]), 1.0)

def test_period_dielectric():
    x = np.linspace(-0.5, 0.5, num=1001)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(x.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)
    
    fdtd = FDTD1D(x, "period", epsilon_vector)

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(time)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(np.abs(R_E[0,1]), 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)

def test_error():
    error = np.zeros(5)
    deltax = np.zeros(5)
    for i in range(5):
        num = 10**(i+1) +1
        x = np.linspace(-0.5, 0.5, num)
        deltax[i] = 1/num
        fdtd = FDTD1D(x, "pec")
        spread = 0.1
        initialE = np.exp( - ((x-0.1)/spread)**2/2)
        
        fdtd.setE(initialE)
        fdtd.step()
        fdtd.step()
        N = len(initialE)
        error[i] = np.sqrt(np.sum((fdtd.getE() - initialE)**2)) / N
        
    # plt.plot(deltax, error)
    # plt.loglog()
    # plt.grid(which='both')
    # plt.show()
    
    # np.polyfit(np.log10(error), np.log10(deltax), 1)
    
    slope = (np.log10(error[-1]) - np.log10(error[0])) / \
        (np.log10(deltax[-1]) - np.log10(deltax[0]) )


    assert np.isclose( slope , 2, rtol=1.e-1)
    

def test_pec_block_dielectric():
    x = np.linspace(-1.0, 1.0, num=201)
    epsilon_r = 4
    inter = 151
    epsilon_vector = np.concatenate((np.ones(inter), epsilon_r*np.ones(x.size-inter)))
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(x, "pec", epsilon_vector)

    spread = 0.1
    initialE = 2*np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(0.75)

    E_left = fdtd.getE()[:101]
    E_right = fdtd.getE()[101:]

    E_left_max = np.max(E_left)
    E_right_max = np.max(E_right)
    E_right_min = np.min(E_right)

    Reflection = np.abs(E_right_min/E_left_max)
    Transmission = np.abs(E_right_max/E_left_max)

    print(Reflection + Transmission)

    assert np.isclose(Reflection + Transmission, 1, atol = 0.004)

    R = np.corrcoef(fdtd.getE(), -initialE)
    # assert np.isclose(R[0,1], 1.0)
