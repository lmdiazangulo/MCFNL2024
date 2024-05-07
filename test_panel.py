# Clase 14/03/2024
# Primer ejemplo de programación con tests

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as sci

EPSILON_0 = 8.85e-12
MU_0 = np.pi * 4 * 1e-7
ETA_0 = np.sqrt(MU_0/EPSILON_0)

class Panel():
    ETA0 = sci.constants
    def __init__(self, eps_r, mu_r, sigma, thickness):
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma = sigma
        self.thickness = thickness

    def get_epsC(self, w):
        self.epsC = self.eps_r * EPSILON_0-1j*self.sigma/w

    @staticmethod
    def gamma(w, epsC, mu):
        return 1j * w *np.sqrt(mu * epsC)
        
    def phi(self, w):
        self.get_epsC(w)
        epsC = self.epsC
        mu = self.mu_r * MU_0
        gd = self.gamma(w, epsC, mu)*self.thickness
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


class Panel_c(Panel):
    def __init__(self, dielectric, thickness, mu_r = 1):

        self.eps_inf = dielectric["eps_inf"]
        self.poles = dielectric["poles"]
        self.residuals = dielectric["residuals"]
        self.sigma = dielectric["sigma"]
        self.thickness = thickness
        self.mu_r = mu_r

    def get_epsC(self, w):
        self.epsC = EPSILON_0*self.eps_inf + EPSILON_0 * (np.sum(self.residuals/(1j*w - self.poles)\
                        + np.conjugate(self.residuals)/(1j*w - np.conjugate(self.poles))))\


    def getTransmissionCoefficient_c(self, w_array):
        dim = len(w_array)
        T = np.zeros(dim)
        for i, w in enumerate(w_array):
            self.get_epsC(w)
            T[i] = self.getTransmissionCoefficient(w)
        return T

    def getReflectionCoefficient_c(self, w_array):
        dim = len(w_array)
        R = np.zeros(dim)
        for i, w in enumerate(w_array):
            R[i] = self.getReflectionCoefficient(w)
        return R



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

def test_void_panel():
    panel = Panel(eps_r =  1.0, mu_r = 1.0, sigma = 0.0, thickness = 1e-3)
    fq = 1e6
    w = 2.0*np.pi*fq
    
    R = panel.getReflectionCoefficient(w)
    T = panel.getTransmissionCoefficient(w)

    assert np.allclose(np.abs(T)**2, 1.0) 


def test_dispersive_panel():
    idx_ini = 150
    idx_fin = 165
    eps_inf = 1  
    sigma = 0     

    residuals = np.array([ 5.987e-1 + 4.195e3j, -2.211e-1 + 2.680e-1j, -4.240 + 7.324e2j,
                           6.391e-1 + 7.186e-2j, 1.806 + 4.563j, 1.443 - 8.219e1j])
    poles = np.array([ -2.502e-2 - 8.626e-3j, -2.021e-1 - 9.407e-1j, -1.467e1 - 1.338j,
                       -2.997e-1 - 4.034j, -1.896 - 4.808j, -9.396 - 6.477j])

    dielectric = {
            "idx_ini":idx_ini,
            "idx_fin":idx_fin,
            "eps_inf":eps_inf,
            "sigma":sigma,
            "poles":poles,
            "residuals":residuals,
        }
    thickness = 0.2
    panel = Panel_c(dielectric, thickness)

    w = np.linspace(1, 1e6, 500)

    R = panel.getReflectionCoefficient_c(w)
    T = panel.getTransmissionCoefficient_c(w)

    assert np.allclose(np.abs(R)**2 + np.abs(T)**2, 1.0, rtol = 1e-3 )