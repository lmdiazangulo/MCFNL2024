# Clase 14/03/2024
# Primer ejemplo de programación con tests

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as sci


EPSILION_0 = 8.85e-12
MU_0 = np.pi * 4 * 1e-7
ETA_0 = np.sqrt(MU_0/EPSILION_0)

class Panel():
    ETA0 = sci.constants
    def __init__(self, eps_r, mu_r, sigma, thickness):
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma = sigma
        self.thickness = thickness
        
    def phi(self, w):
        mu = self.mu_r * MU_0
        epsC = self.eps_r * EPSILION_0-1j*self.sigma/w
        gamma = 1j * w *np.sqrt(mu * epsC)
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



# Primero escribimos lo que queremos que haga el programa, no lo que tiene que tener "debajo"
# Cómo nos gustaría poder usarlo? Qué tiene que hacer bien para que me crea que está funcionando?
# Los test deben tener ninguno o muy pocos argumentos.

def test_non_power_dissipation():
    # Un test siempre tiene tres partes:
    
    # 1.- Cuando tengo algo
    panel = Panel(eps_r =  3.0, mu_r = 1.0, sigma = 0.0, thickness = 1e-3)
    fq = 1e6
    w = 2.0*np.pi*fq
    
    # 2.- Y haga algo
    R = panel.getReflectionCoefficient(w)
    T = panel.getTransmissionCoefficient(w)

    # 3.- Obtengo algo
    assert np.allclose(np.abs(R)**2 + np.abs(T)**2, 1.0) # Assert comprueba que la condición es cierta y si no salta error
