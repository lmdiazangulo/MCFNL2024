import numpy as np
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

class FDTD1D():
    def __init__(self, xE, boundary, relative_epsilon_vector=None, sigma_vector = None):
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0


        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = xE[1] - xE[0]
        self.dt = 1.0 * self.dx

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

        epsilon = self.epsilon_r * EPSILON_0
        E_factor = (epsilon[1:-1] - self.sigma_vector[1:-1]*self.dt/2)\
            /(epsilon[1:-1] + self.sigma_vector[1:-1]*self.dt/2)
        H_factor = self.dt/(epsilon[1:-1] + self.sigma_vector[1:-1]*self.dt/2) * (1/self.dx)
        E[1:-1] =  E_factor * E[1:-1] - H_factor * (H[1:] - H[:-1])
        # E[1:-1] += - c_eps[1:-1] * (H[1:] - H[:-1])
        for source in self.sources:
            E[source.location] += source.function(self.t)
        self.t += self.dt
        H += - c0 *(E[1:] - E[:-1])
        for source in self.sources:
            H[source.location] += source.function(self.t)

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c0 / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c0 / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        else:
            raise ValueError("Boundary not defined")

    def run_until(self, finalTime):
        while (self.t <= finalTime):
            if False:    
                plt.plot(self.xE, self.E, '.-')
                # plt.plot(self.xH, self.H, '.-')
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
    assert np.isclose(R[0,1], 1.0)

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

    fdtd.addSource(Source.gaussian(20, 0.5, 0.5, 0.1))
    fdtd.addSource(Source.gaussian(70, 1.0, -0.5, 0.1))

    fdtd.run_until(1.0)
    assert np.allclose(fdtd.getE()[:20], 0.0, atol = 1e-2)
    assert np.allclose(fdtd.getE()[71:], 0.0, atol = 1e-2)
    fdtd.run_until(3.0)
    assert np.allclose(fdtd.getE(), 0.0, atol = 1e-2)

def test_conductivity_absorption():
    x = np.linspace(-0.5, 0.5, num = 101) 
    sigma_vector = np.zeros(x.size)
    sigma_vector[:] = 1.0
    fdtd = FDTD1D(x, "pec", sigma_vector = sigma_vector)

    fdtd.addSource(Source.gaussian(20, 0.5, 0.5, 0.05))
    fdtd.run_until(0.4)
    energy0 = np.sum(fdtd.getE())
    fdtd.run_until(10.5)
    energy1 = np.sum(fdtd.getE())
    assert energy1 < energy0
