import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

class FDTD1D():
    def __init__(self, xE, boundary):
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0

        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = xE[1] - xE[0]
        self.dt = 1.0 * self.dx
        self.t = 0

        self.boundary = boundary


    def add_illumination(self, distribution, t):
        distribution(t)
        pass

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
        c = self.dt/self.dx

        spread = 0.1
        E[1:-1] += - c * (H[1:] - H[:-1])
        initialE = np.exp( - ((self.t-0.5)/spread)**2/2)
        self.E[20] += initialE
        initialE = np.exp( - ((self.t-0.5)/spread)**2/2)
        self.E[20] += initialE
        self.t += self.dt/2

        H += - c * (E[1:] - E[:-1])
        initialH = np.exp( - ((self.t-0.5+self.dt/2)/spread)**2/2)
        self.H[20] += initialH
        self.t += self.dt/2

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c / EPSILON_0 * (2 * H[0])
            E[-1] = E[-1] - c / EPSILON_0 * (-2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c * (H[0] - H[-1])
            E[-1] = E[0]
        else:
            raise ValueError("Boundary not defined")

    def run_until(self, finalTime):
        t = 0.0
        
        while (t < finalTime):

            plt.plot(self.xE, self.E, '.-')
            plt.ylim(-1.1, 1.1)
            plt.grid(which='both')
            plt.pause(0.001)
            plt.cla()
            
            self.step()
            # t += self.dt

        


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
    y = (x[1:] - x[:-1]) / 2
    fdtd = FDTD1D(x, "pmc")

    spread = 0.1
    initialH = np.exp( - (y/spread)**2/2)

    fdtd.setH(initialH)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getH(), -initialH)
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
    
    x = norm(1, 2)
    print(x)
    
    plt.plot(x)
    plt.show()
    # x = np.linspace(-0.5, 0.5, 101)
    # x_sub = x[30:60]
    # y = (x[1:] - x[:-1]) / 2
    # fdtd = FDTD1D(x, "pec") 
    
    # fdtd.run_until(2.0)
    # fdtd.step()
    
    # E = fdtd.getE()
    
 
def test_total_field_scattered_field():
     
    # x = np.linspace()
    return
    
def g():
    return 0

    
    
     