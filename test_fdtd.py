import numpy as np
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

class FDTD1D():
    def __init__(self, xE):
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0

        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = xE[1] - xE[0]
        self.dt = 1.0 * self.dx

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

    def step_fdtd(self):
        E = self.E
        H = self.H
        c = self.dt/self.dx

        H += - c * (E[1:] - E[:-1])
        E[1:-1] += - c * (H[1:] - H[:-1])
        E[0] = 0.0
        E[-1] = 0.0

    def step_pmc(self):
        E = self.E
        H = self.H
        c = self.dt/self.dx

        H += - c * (E[1:] - E[:-1])
        E[1:-1] += - c * (H[1:] - H[:-1])
        E[0] = E[0] - c / EPSILON_0 * (2 * H[0])
        E[-1] = E[-1] - c / EPSILON_0 * (-2 * H[-1])


    def run_until(self, finalTime, is_fdtd = True):
        t = 0.0
        

        if (is_fdtd):

            while (t < finalTime):
                
                #plt.plot(self.xE, self.E, '.-')
                #plt.ylim(-1.1, 1.1)
                #plt.grid(which='both')
                #plt.pause(0.001)
                #plt.cla()
                
                self.step_fdtd()
                t += self.dt

        else:

            while (t < finalTime):
                
                #plt.plot(self.xE, self.E, '.-')
                #plt.ylim(-1.1, 1.1)
                #plt.grid(which='both')
                #plt.pause(0.001)
                #plt.cla()
                
                self.step_pmc()
                t += self.dt


def test_fdtd():
    x = np.linspace(-0.5, 0.5, num=101)
    fdtd = FDTD1D(x)

    spread = 0.1
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(R[0,1], 1.0)

def test_pmc():
    x = np.linspace(-0.5, 0.5, num=101)
    y = (x[1:] - x[:-1]) / 2
    fdtd = FDTD1D(x)

    spread = 0.1
    initialH = np.exp( - (y/spread)**2/2)

    fdtd.setH(initialH)
    fdtd.run_until(1.0, False)

    R = np.corrcoef(fdtd.getH(), -initialH)
    assert np.isclose(R[0,1], 1.0)

