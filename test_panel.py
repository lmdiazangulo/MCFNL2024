import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as sci

EPSILON_0 = 8.854e-12
MU_0 = np.pi * 4 * 1e-7
ETA0 = np.sqrt(MU_0 / EPSILON_0)

class Panel():
    def __init__(self, eps_r, mu_r, sigma, thickness):
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma = sigma
        self.thickness = thickness

    def phi(self, w):
        mu = self.mu_r * MU_0
        epsC = self.eps_r * EPSILON_0 - 1j * self.sigma / w
        gamma = 1j * w * np.sqrt(mu * epsC)
        gd = gamma * self.thickness
        eta = np.sqrt( mu / epsC )
        return np.array(
            [ [      np.cosh(gd), eta * np.sinh(gd)], 
              [1/eta*np.sinh(gd),       np.cosh(gd)],]
        )
    
    def denominator(self, w):
        return (self.phi(w)[0,0] * ETA0 + self.phi(w)[0,1] + self.phi(w)[1,0]*ETA0**2 + self.phi(w)[1,1]*ETA0) 

    def getTransmissionCoefficient(self, w):
        return 2*ETA0 / self.denominator(w)

    def getReflectionCoefficient(self, w):
        return (self.phi(w)[0,0] * ETA0 + self.phi(w)[0,1] - self.phi(w)[1,0]*ETA0**2 - self.phi(w)[1,1]*ETA0) / \
              self.denominator(w) 


def test_non_power_dissipation():
    panel = Panel(eps_r=3.0, mu_r=1.0, sigma=0.0, thickness=1e-3)

    fq = 1e6
    w = 2.0 * np.pi * fq
    R = panel.getReflectionCoefficient(w)
    T = panel.getTransmissionCoefficient(w)

    assert np.allclose(np.abs(R)**2 + np.abs(T)**2, 1.0)