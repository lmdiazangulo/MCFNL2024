import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

class FDTD1D():
    def __init__(self, xE, boundary, dielectric=None, relative_epsilon_vector=None):
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0

        if dielectric is None:
            self.diec_ex = 0
        else:
            self.diec_ex = 1
            self.dielectric = dielectric
            self.J = np.zeros((self.dielectric["poles"].shape[0], self.xE.shape[0]))

        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = xE[1] - xE[0]
        self.dt = 1 * self.dx

        self.sources = []
        self.t = 0.0

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

        if self.diec_ex == 1:
            J = self.J
            idx_ini = self.dielectric["idx_ini"]
            idx_fin = self.dielectric["idx_fin"]
            eps_inf = self.dielectric["eps_inf"]
            sigma = self.dielectric["sigma"]
            poles = self.dielectric["poles"]
            residuals = self.dielectric["residuals"]

            # H_old_1 = H[idx_ini+1:idx_fin-1].copy()
            # H_old_2 = H[idx_ini:idx_fin-2].copy()
            H_old = np.zeros_like(H)
            H_old = H
            # E_old = np.zeros_like(E[idx_ini+1:idx_fin-1])
            # E_old = E[idx_ini+1:idx_fin-1]
            E_old = E[idx_ini+1:idx_fin-1].copy()
            pass

        c = self.dt/self.dx
        c_eps = np.ones(self.epsilon_r.size)
        c_eps[:] = self.dt/self.dx / self.epsilon_r[:]
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        H += - self.dt/self.dx *(E[1:] - E[:-1])
        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)
        
        E[1:-1] += - c_eps[1:-1] * (H[1:] - H[:-1])
        for source in self.sources:
            E[source.location] += source.function(self.t + self.dt - self.dx/2)
        self.t += self.dt
        
        if self.diec_ex == 1:
            k = (1 + poles*self.dt/2) / (1 - poles*self.dt/2)
            beta = (EPSILON_0 * residuals* self.dt) / (1 - poles*self.dt/2)
            
            aux = 2 * EPSILON_0 * eps_inf + np.sum(2* np.real(beta))
            aux2 = sigma * self.dt

            E_new = (aux - aux2)/(aux + aux2)* E_old \
                    - 2 * self.dt * ((H_old[idx_ini+1:idx_fin-1] - H_old[idx_ini:idx_fin-2])/self.dx \
                    + np.real(np.sum((1+k[:,np.newaxis]) * J[:, idx_ini+1:idx_fin-1], axis=0)))/(aux + aux2)
            
            for i in range(poles.shape[0]):
                J[i, idx_ini+1:idx_fin-1] = k[i]*J[i, idx_ini+1:idx_fin-1] + beta[i] * (E_new - E_old)/self.dt

            E[idx_ini+1:idx_fin-1] = E_new
            H = H_old - self.dt/self.dx * (E[1:] - E[:-1])


        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        elif self.boundary == "mur":
            cte = (c-1.0)/(c + 1.0)
            # Left 
            E[0] = E_aux_izq + cte*( E[1] - E[0])
            # Right
            E[-1] = E_aux_dch + cte*( E[-2] - E[-1] )
        else:
            raise ValueError("Boundary not defined")

    def run_until(self, finalTime):
        while (self.t <= finalTime):
            if False:
                if self.diec_ex == 1:
                    plt.vlines([self.xE[self.dielectric["idx_ini"]],self.xE[self.dielectric["idx_fin"]-1]],[-1,-1],[1,1], color='red')    
                    plt.axvspan(self.xE[self.dielectric["idx_ini"]],self.xE[self.dielectric["idx_fin"]-1], color='gray', alpha=0.5)  # Desde x=1 hasta x=5
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
            return np.exp( - ((t-center)/spread)**2/2) * amplitude
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

    fdtd = FDTD1D(x, "pec", relative_epsilon_vector=epsilon_vector)

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
    
    fdtd = FDTD1D(x, "period", relative_epsilon_vector=epsilon_vector)

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
    fdtd = FDTD1D(x, "pec",dielectric=None, relative_epsilon_vector=None)
    finalTime = 1.0

    fdtd.addSource(Source.gaussian(20, 0.5, 0.5, 0.1))
    fdtd.addSource(Source.gaussian(70, 1.0, -0.5, 0.1))

    while (fdtd.t <= finalTime):
        # plt.plot(fdtd.xE, fdtd.E, '.-')
        # plt.plot(fdtd.xH, fdtd.H, '.-')
        # plt.ylim(-1.1, 1.1)
        # plt.title(fdtd.t)
        # plt.grid(which='both')
        # plt.pause(0.02)
        # plt.cla()
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

    fdtd = FDTD1D(x, "pec", relative_epsilon_vector=epsilon_vector)

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
    
    fdtd = FDTD1D(x, "period", relative_epsilon_vector=epsilon_vector)

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(time)


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
    

def test_pec_block_dielectric():
    x = np.linspace(-1.0, 1.0, num=201)
    epsilon_r = 4
    inter = 151
    epsilon_vector = np.concatenate((np.ones(inter), epsilon_r*np.ones(x.size-inter)))
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(x, "pec", relative_epsilon_vector = epsilon_vector)

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

def test_dispersive_null_panel():
    num=201
    
    idx_ini = 150
    idx_fin = num - 10
    eps_inf = 1  #Buscar que valores poner
    sigma = 0    #Buscar que valores poner
    poles = np.array([ 0, 0, 0, 0])
    residuals = np.array([ 0, 0, 0, 0])

    dielectric = {
            "idx_ini":idx_ini,
            "idx_fin":idx_fin,
            "eps_inf":eps_inf,
            "sigma":sigma,
            "poles":poles,
            "residuals":residuals,
        }

    x = np.linspace(-0.5, 0.5, num)
    fdtd1 = FDTD1D(x, "pec", dielectric = dielectric)
    fdtd2 = FDTD1D(x, "pec", dielectric = None)

    dielectric2 = dielectric.copy()
    dielectric2["eps_inf"] = 2.3
    dielectric2["idx_ini"] = 0
    dielectric2["idx_fin"] = num
    fdtd3 = FDTD1D(x, "pec", dielectric=dielectric2)
    fdtd4 = FDTD1D(x, "pec", relative_epsilon_vector=np.ones(num)*2.3, dielectric=None)
    
    spread = 0.01
    initialE = np.exp( - (x/spread)**2/2)

    fdtd1.setE(initialE)
    fdtd2.setE(initialE)
    fdtd3.setE(initialE)
    fdtd4.setE(initialE)

    fdtd1.run_until(0.25)   
    fdtd2.run_until(0.25)   
    fdtd3.run_until(0.25)   
    fdtd4.run_until(0.25)   

    assert np.allclose(fdtd1.getE(), fdtd2.getE(), rtol = 1e-8)
    assert np.allclose(fdtd1.getH(), fdtd2.getH(), rtol = 1e-8)

    assert np.allclose(fdtd3.getE(), fdtd4.getE(), rtol = 1e-8)
    assert np.allclose(fdtd3.getH(), fdtd4.getH(), rtol = 1e-8)


def test_dispersive_as_pec():
    # Definimos polos y residuos 0 con sigma tendiendo a infinito y el comportamiento es como en un pec.
    num = 101
    idx_ini = num -1
    idx_fin = num + 10
    eps_inf = 1  #Buscar que valores poner
    sigma = 1e10    #Buscar que valores poner
    residuals = np.zeros(5)
    poles = np.zeros(5)

    dielectric = {
        "idx_ini":idx_ini,
        "idx_fin":idx_fin,
        "eps_inf":eps_inf,
        "sigma":sigma,
        "poles":poles,
        "residuals":residuals,
    }

    num = 101
    x1 = np.linspace(-0.5, 0.5, num=num)
    x2 = np.concatenate((x1[:-1], np.linspace(0.5, 1.0, num = num)))

    fdtd = FDTD1D(x2, "pec", dielectric=dielectric)

    spread = 0.05
    initialE = np.zeros(len(x2))
    initialE[:num] = np.exp( - (x1/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.01)

    R = np.corrcoef(fdtd.getE(), -initialE)

    # assert np.allclose(-fdtd.getE(), initialE)
    
    assert np.isclose(R[0,1], 1.0, rtol = 0.015)

def test_dispersive_panel():
    num=201
    
    idx_ini = 150
    idx_fin = num - 10
    eps_inf = 1  #Buscar que valores poner
    sigma = 1e10    #Buscar que valores poner

    residuals = np.array([ 5.987e-1 + 4.195e3j, -2.211e-1 + 2.680e-1j, -4.240 + 7.324e2j,
                          6.391e-1 + 7.186e-2j, 1.806 + 4.563j, 1.443 - 8.219e1j])
    poles = np.array([ -2.502e-2 - 8.626e-3j, -2.021e-1 - 9.407e-1j, -1.467e1 - 1.338j,
                      -2.997e-1 - 4.034j, -1.896 - 4.808j, -9.396 - 6.477j])
    
    residuals = np.zeros(len(residuals))
    poles = np.zeros(len(poles))

    dielectric = {
            "idx_ini":idx_ini,
            "idx_fin":idx_fin,
            "eps_inf":eps_inf,
            "sigma":sigma,
            "poles":poles,
            "residuals":residuals,
        }

    x = np.linspace(-0.5, 0.5, num)
    fdtd = FDTD1D(x, "pec", dielectric = dielectric)
    
    spread = 0.01
    initialE = np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)
    # fdtd.addSource(Source.gaussian(90, 0.5, 0.5, 0.1))
    fdtd.run_until(0.75)
    
    # Test con refelxión y trnasmisión como el panel.
    
    E_left = fdtd.getE()[:101]
    E_right = fdtd.getE()[101:]
    
    E_left_max = np.max(E_left)
    E_right_max = np.max(E_right)
    E_right_min = np.min(E_right)

    Reflection = np.abs(E_right_min/E_left_max)
    Transmission = np.abs(E_right_max/E_left_max)

    from test_panel import Panel_c

    panel = Panel_c(w)
    fq = 1e6
    w = 2.0*np.pi*fq
    
    R = panel.getReflectionCoefficient_c(w)
    T = panel.getTransmissionCoefficient_c(w)

    assert np.isclose(Reflection, R)
    assert np.isclose(Transmission, T)

    
    E_left = fdtd.getE()[:101]
    E_right = fdtd.getE()[101:]
    
    E_left_max = np.max(E_left)
    E_right_max = np.max(E_right)
    E_right_min = np.min(E_right)

    Reflection = np.abs(E_right_min/E_left_max)
    Transmission = np.abs(E_right_max/E_left_max)

    print(Reflection + Transmission)
    
    assert  Reflection + Transmission < 1