import numpy as np

def quadIF_func(V,Vrst,Vth,Gqif,Vr,Vc,C,Iapp):
    return (Gqif*(V-Vr)*(V-Vc)+Iapp)/C

def LIF_func(V,Vrst,Vth,Gl,El,C,Iapp):
    return (-Gl*(V-El)+Iapp)/C

def ExpIF_func(V,Vrst,Vth,Vr,Geif,DeltaT,Vrh,Iapp):
    return -Geif*(V-Vr)+Geif*DeltaT*np.exp((V-Vrh)/DeltaT)+Iapp

def integrate_quadIF(dt, t, V0, Vrst, Vth, Gqif, Vr, Vc, C, Iapp):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = quadIF_func(V[j],Vrst,Vth,Gqif,Vr,Vc,C,Iapp) #(Gqif*(V[j]-Vr)*(V[j]-Vc)+Iapp)/C
        av = V[j]+kv1*dt
        kv2 = quadIF_func(av,Vrst,Vth,Gqif,Vr,Vc,C,Iapp) #(Gqif*(av-Vr)*(av-Vc)+Iapp)/C
        V[j+1] = V[j] + (kv1+kv2)*dt/2
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)

def integrate_quadIF_noise(dt, t, V0, Vrst, Vth, Gqif, Vr, Vc, C, Iapp, Ddt):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = quadIF_func(V[j],Vrst,Vth,Gqif,Vr,Vc,C,Iapp)#(Gqif*(V[j]-Vr)*(V[j]-Vc)+Iapp)/C
        av = V[j]+kv1*dt
        kv2 = quadIF_func(av,Vrst,Vth,Gqif,Vr,Vc,C,Iapp)#(Gqif*(av-Vr)*(av-Vc)+Iapp)/C
        V[j+1] = V[j] + (kv1+kv2)*dt/2 + np.sqrt(2.0*Ddt)*np.random.randn()
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)

def integrate_quadIF_adapt(dt, t, V0, g0, Vrst, Vth, Gqif, Vr, Vc, C, Ek, tau, Deltag, Iapp):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    g = np.zeros(shape=t.shape)
    V[0] = V0
    g[0] = g0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kg1 = -g[j]/tau
        kv1 = (Gqif*(V[j]-Vr)*(V[j]-Vc)+Iapp-g[j]*(V[j]-Ek))/C
        av = V[j]+kv1*dt
        ag = g[j]+kg1*dt
        kg2 = -ag/tau
        kv2 = (Gqif*(av-Vr)*(av-Vc)+Iapp-ag*(av-Ek))/C
        g[j+1] = g[j] + (kg1+kg2)*dt/2
        V[j+1] = V[j] + (kv1+kv2)*dt/2
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            g[j+1] = g[j] + Deltag
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,g,spkcnt,tspk)

def integrate_LIF(dt, t, V0, Vrst, Vth, Gl, El, C, Iapp):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = LIF_func(V[j],Vrst,Vth,Gl,El,C,Iapp)# (-Gl*(V[j]-El)+Iapp)/C
        av = V[j]+kv1*dt
        kv2 = LIF_func(av,Vrst,Vth,Gl,El,C,Iapp)#(-Gl*(av-El)+Iapp)/C
        V[j+1] = V[j] + (kv1+kv2)*dt/2
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)

def integrate_LIF_noise(dt, t, V0, Vrst, Vth, Gl, El, C, Iapp, Ddt):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = LIF_func(V[j],Vrst,Vth,Gl,El,C,Iapp) #(-Gl*(V[j]-El)+Iapp)/C
        av = V[j]+kv1*dt
        kv2 = LIF_func(av,Vrst,Vth,Gl,El,C,Iapp) #(-Gl*(av-El)+Iapp)/C
        V[j+1] = V[j] + (kv1+kv2)*dt/2 + np.sqrt(2.0*Ddt)*np.random.randn()
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)

def integrate_expIF(dt, t, V0, Vrst, Vth, Vr, Geif, DeltaT, Vrh, Iapp):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = ExpIF_func(V[j],Vrst,Vth,Vr,Geif,DeltaT,Vrh,Iapp)#-Geif*(V[j]-Vr)+Geif*DeltaT*np.exp((V[j]-Vrh)/DeltaT)+Iapp
        av = V[j]+kv1*dt
        kv2 = ExpIF_func(av,Vrst,Vth,Vr,Geif,DeltaT,Vrh,Iapp)#-Geif*(av-Vr)+Geif*DeltaT*np.exp((av-Vrh)/DeltaT)+Iapp
        V[j+1] = V[j] + (kv1+kv2)*dt/2
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)

def integrate_expIF_noise(dt, t, V0, Vrst, Vth, Vr, Geif, DeltaT, Vrh, Iapp, Ddt):
    # t -> time vector
    # V -> V vector
    V = np.zeros(shape=t.shape)
    V[0] = V0
    spkcnt = 0
    tspk = []
    for j in range(len(t)-1):
        kv1 = ExpIF_func(V[j],Vrst,Vth,Vr,Geif,DeltaT,Vrh,Iapp)#-Geif*(V[j]-Vr)+Geif*DeltaT*np.exp((V[j]-Vrh)/DeltaT)+Iapp
        av = V[j]+kv1*dt
        kv2 = ExpIF_func(av,Vrst,Vth,Vr,Geif,DeltaT,Vrh,Iapp)#-Geif*(av-Vr)+Geif*DeltaT*np.exp((av-Vrh)/DeltaT)+Iapp
        V[j+1] = V[j] + (kv1+kv2)*dt/2 + np.sqrt(2.0*Ddt)*np.random.randn()
        if V[j+1]>=Vth:
            V[j] = 60.0
            V[j+1] = Vrst
            spkcnt = spkcnt+1
            tspk.append(t[j])
    return (V,spkcnt,tspk)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def get_current_ramp(Imax,nTimeSteps):
    nTimeSteps = np.abs(nTimeSteps)
    if nTimeSteps > 1:
        a = Imax / float(nTimeSteps)
    else:
        return Imax
    return np.asarray([ a*float(t+1) for t in range(int(nTimeSteps)) ])


def get_input_current(ts,dt,stimtype,I0,tStim,DeltaTStim):
    assert stimtype.lower() in ['step','ramp'],'stimtype must be either step or ramp'
    tIext  = int(np.ceil(tStim / dt))
    dtIext = int(np.ceil(DeltaTStim / dt))
    I = np.zeros(len(ts))
    if stimtype.lower() == 'step':
        I[np.logical_and(ts >= tIext, ts < (tIext+dtIext))] = I0
    elif stimtype.lower() == 'ramp':
        I[np.logical_and(ts >= tIext, ts < (tIext+dtIext))] = get_current_ramp(I0,dtIext)
    return I

class SynapticInput:
    def __init__(self):
        return
    def GetSignal(self):
        return 0.0

class SynapticWhiteNoise(SynapticInput):
    def __init__(self,stddev,dt,mean=0.0):
        stddev = stddev / 2.0
        self.sqrt_D_dt = np.sqrt(stddev*stddev*dt)
        self.mean = mean
    
    def GetSignal(self):
        return self.mean + self.sqrt_D_dt * np.random.normal()

class LIF:
    def __init__(self,dt,VInit,Vr,Vb,tau,R,theta,noiseSignal=None,setIC=True):
        self.Vrec = VInit
        self.VInit = VInit
        self.Vr = Vr
        self.Vb = Vb
        self.theta = theta
        self.invTau = 1.0 / tau
        self.R = R
        self.AddNoiseFunc(noiseSignal)
        self.dV1 = 0.0
        self.dV2 = 0.0
        self.dt = dt
        self.input = []
        if setIC:
            self.SetIC(VInit=VInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        r     = self.noiseSignal()
        self.dV1 = (self.R*Iext - self.V + self.Vb)*self.invTau
        self.dV2 = (self.R*Iext - (self.V + self.dt*self.dV1 + r) + self.Vb)*self.invTau
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + r

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetV(self):
        return self.Vrec
    
    def GetThreshold(self):
        return self.theta
    
    def AddInput(self,inp):
        self.input.append(inp)
    
    def AddNoiseFunc(self,noiseSignalFunc):
        if noiseSignalFunc:
            self.noiseSignal = noiseSignalFunc
        else:
            self.noiseSignal = lambda x:0.0

    def SumInput(self):
        s = 0.0
        for I in self.input:
            s += I.GetSignal()
        return s
    
    def Reset(self):
        self.V  = self.VInit
        self.Vrec = self.V

    def SetAttrib(self,**kwargs):
        self.__dict__.update(kwargs)
        if any(['init' in s.lower() for s in kwargs.keys()]):
            self.Reset()

    def GetStateAsIC(self):
        return dict(VInit=self.V)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext)
    
    def SetIC(self,**kwargs):
        if any([np.isnan(v) for v in kwargs.values()]):
            d = self.GetRestingState()
            self.SetAttrib(**dict([ ( (k[0] if k == 'gK' else k)+'Init',v) for k,v in d.items() ]))
        else:
            self.SetAttrib(**dict([ ( (k[0] if k == 'gK' else k)+('' if 'init' in k.lower() else 'Init' ),v) for k,v in kwargs.items() ]))