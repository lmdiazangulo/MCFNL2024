import numpy as np
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

class FDTD1D():
    def __init__(self, xE, boundary, relative_epsilon_vector=None, s_e = None, s_m=None):
        
        self.xE = xE
        self.xH = (xE[1:] + xE[:-1]) / 2.0

        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = xE[1] - xE[0]
        self.dt = 1.0 * self.dx

        self.sources = []
        self.t = 0.0

        if relative_epsilon_vector is None:
            self.epsilon_r = np.ones(self.xE.shape)
        else:
            self.epsilon_r = relative_epsilon_vector
        
        if s_e is None:
            self.sigma_e = np.zeros(self.xE.shape)
        else:
            self.sigma_e = s_e[:]

        if s_m is None:
            self.sigma_m = np.zeros(self.xH.shape)
        else:
            self.sigma_m = s_m[:]
        
        self.mu_r = np.ones_like(self.xH)

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
        c = self.dt/self.dx
        
        c_eps = np.ones(self.epsilon_r.size)
        c_eps[:] = self.dt/self.dx / (self.epsilon_r[:] + self.sigma_e[:] * self.dt / 2.)
        
        c_eps2 = np.ones(self.epsilon_r.size)#ATENTO A DIMENSIONES AQUI
        c_eps2[:] = (self.epsilon_r[:] - self.sigma_e[:] * self.dt / 2.) #NUMERADOR

        c_eps3 = np.ones(self.epsilon_r.size)
        c_eps3[:] = (self.epsilon_r[:] + self.sigma_e[:] * self.dt / 2.) #DENOMINADOR 

        c_mu = np.ones(self.mu_r.size)
        c_mu2 = np.ones(self.mu_r.size)

        c_mu[:] = ( self.mu_r + self.sigma_m * self.dt / 2. ) # DENOMINADOR
        c_mu2[:] = (self.mu_r - self.sigma_m * self.dt / 2.)  # NUMERADOR       
        
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        H[:] = H[:] *  (c_mu2 / c_mu) - self.dt/self.dx *(E[1:] - E[:-1]) / c_mu
        
        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)
        
        E[1:-1] += - c_eps[1:-1] * (H[1:] - H[:-1])
        
        for source in self.sources:
            E[source.location] += source.function(self.t + self.dt - self.dx/2)
            
        self.t += self.dt
        
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
            if True:    
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

def test_energy_conservation():
    x = np.linspace(-1, 1, num=201)
    dx = np.abs(x[1] - x[2])
    epsilon_r = 1
    conductivity = 1.0
    mu = 1
    epsilon_vector = epsilon_r * np.ones(x.size)
    conductivity_vector = conductivity * np.ones(x.size)
    fdtd = FDTD1D(x, "pec", epsilon_vector, conductivity_vector)

    spread = 0.1
    initialE = np.exp( - ((x-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)

    fdtd.setE(initialE)
    fdtd.setH(initialH)

    def energy(E, H):
        electric = epsilon_r * np.sum([campo**2 for campo in E])  
        magnetic = mu * np.sum([campo**2 for campo in H])  
        return (electric + magnetic) * dx

    initial_energy = energy(initialE, initialH)
    fdtd.run_until(1)
    final_energy = energy(fdtd.getE(), fdtd.getH()) 

    assert np.isclose(initial_energy, final_energy, atol=1e-3)
    
def test_pml():
    x = np.linspace(-1.0, 1.0, num=201)
    inter = 101
    pml_size = 50
    dx = np.abs(x[1] - x[2])

    epsilon_r = 1
    sigma_e_max = 7

    #sigma_m = sigma_e_max * MU_0 / EPSILON_0
    
    k = (sigma_e_max/(pml_size-1)**2)
    perfil_parabolico_e = [k*(a-pml_size+1)**2 for a in range(pml_size)]
    perfil_parabolico_invertido_e = perfil_parabolico_e[::-1]
    
    #epsilon_vector = np.ones_like(x)
    epsilon_vector = np.concatenate((epsilon_r*np.ones(pml_size), 1.0*np.ones(inter), epsilon_r*np.ones(pml_size)))
    
    sigma_e_vector = np.concatenate((perfil_parabolico_e, 1.0*np.zeros(inter), perfil_parabolico_invertido_e))
    sigma_m_vector = (sigma_e_vector[1:]+sigma_e_vector[-1:])*0.5* MU_0 / EPSILON_0

    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(x, "pec", epsilon_vector, sigma_e_vector, sigma_m_vector)

    spread = 0.1
    initialE = 2*np.exp( - (x/spread)**2/2)

    fdtd.setE(initialE)

    def energy(E, H):
        electric = epsilon_r * np.sum([campo**2 for campo in E])  
        magnetic = MU_0 * np.sum([campo**2 for campo in H])  
        return (electric + magnetic) * dx

    initial_energy = energy(initialE, fdtd.getH())

    fdtd.run_until(2)

    final_energy = energy(fdtd.getE(), fdtd.getH()) 

    assert final_energy/initial_energy < 1 