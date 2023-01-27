import numpy as n
import matplotlib.pyplot as plt
import scipy.special as ss
import stuffr

#
# The ambipolar diffusion-rate is given in units of m/s, but this is not correct.
# I need to change this to the closed form solution of the diffusion equation
# for a gaussian. also, I need to specify the initial radius based on the mean free path. 
#
def efield(v=20e3,lam=6.0, D=20e3, L=20e3, N=2000, dec=10, diff_rate=3.0,mean_free_path=0.2):
    T_max=L/v
    dt=T_max/N
    x0=L/2.0
    E=n.zeros(N,dtype=n.float32)
    print(N)

    dx=dt*v
    t=n.arange(N)*dt
    x=t*v
    
    dists=n.sqrt(D**2.0 + (x-x0)**2.0)
    
    Es=n.exp(1j*(4.0*n.pi*dists/lam))/dists

    P=n.zeros([N,len(x)],dtype=n.complex64)
    P2=n.zeros([int(N/dec),int(len(x)/dec)],dtype=n.complex64)
    xdec=n.zeros(int(len(x)/dec))

    E=n.zeros(N,dtype=n.complex64)
    for i in range(2,N):
        # trail lifetime
        trail_lifetime = dt*i - n.arange(i)*dt
        # trail width
        trail_width = diff_rate*trail_lifetime + mean_free_path
        scale=gaussian_integral(width=trail_width, k=4.0*n.pi/lam)
        E[i]=n.sum(Es[0:i]*scale)*dx
        P[i,0:i]=Es[0:i]*scale*dx

    t=t-n.mean(t)
    idx=n.arange(int(N/dec),dtype=int)
    for i in range(dec):
        P2+=stuffr.decimate(P[:,idx*dec+i],dec=dec)
        xdec+=x[idx*dec+i]
    xdec=xdec/dec
    tdec=stuffr.decimate(t,dec=dec)
    return(t,E,P2,xdec,tdec)

def gaussian_integral(width=10.0, k=4.0*n.pi/6.0):
    """ fourier transform of a Gaussian """

    kp=k/n.pi/2.0
    
    a=1.0/(2*width**2.0)
    return((1.0/(n.sqrt(2.0*n.pi*width**2.0)))*n.sqrt(n.pi/a)*n.exp(-n.pi**2.0*kp**2.0/a))

# 
# Plot the distribution of power scattered from different parts of the trail path as a function of time and position
# 
t10,E10,P10,xd,td=efield(v=10e3,N=40000,dec=100,lam=6,diff_rate=3)
dB=10.0*n.log10(n.abs(P10)**2.0)
dB_max=n.nanmax(dB)
dB[n.isinf(dB)]=-999
plt.pcolormesh(xd-n.mean(xd),td,dB-dB_max,vmin=-20,vmax=0)

plt.colorbar()
plt.title("Power (dB)")
plt.xlabel("Along trajectory distance (m)")
plt.ylabel("Time (s)")
plt.show()


plt.plot(t10,n.abs(E10)**2.0)
plt.show()

t20,E20,P20,x,t=efield(v=20e3,N=8000,lam=6)
t70,E70,P70,x,t=efield(v=70e3,N=8000,lam=6)

plt.plot(t10,n.abs(E10)**2.0)
plt.plot(t20,n.abs(E20)**2.0)
plt.plot(t70,n.abs(E70)**2.0)
plt.xlabel("Time (s)")
plt.ylabel("Power (linear)")
plt.show()
        
