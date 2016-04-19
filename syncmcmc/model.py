from priorclasses import FluxFrequencyPriors, UniformPrior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import emcee
import corner
import os
import argparse
from PTmcmc import run_mcmc



"""
   Run a Markov-Chain Monte Carlo sampler to determine best fit parameters for a synchrotron model. 


   Usage: model.py [options] 

   Options:

   -i --input        Specify data file
   -r --raw          Plot raw data

   
   Fit produces F_v, v_a, v_m 
   
   F_v               ::  Flux normalization (F_v(v=v_a))
   v_a               ::  Self-absorption frequency
   v_m               ::  Peak frequency
   
   Known Parameters:
   
   z                 ::  Source redshift
   p                 ::  Power law index of electron Lorentz factor distribution
   epsilon_e         ::  Fraction of energy in electrons as a function of p
   epsilon_b         ::  Fraction of energy in the magnetic field
   d                 ::  Luminosity distance
   beta_1            ::  Spectral slope below break 1
   beta_2            ::  Spectral slope below break 2
   beta_3            ::  Spectral slope below break 3
   s_1               ::  Shape of spectrum at break 1
   s_2               ::  Shape of spectrum at break 2
   s_3               ::  Shape of spectrum at break 3
"""


F_true = 10**(0.96)
va_true = 10**(10.11)
vm_true = 10**(11.41)

# Define known parameters

p = 2.5
epsilon_e = 0.1 * (p-2.)/(p-1.)
epsilon_b = 0.1


# Spectral slopes

beta_1 = 2.
beta_2 = 1./3.
beta_3 = (1.-p)/2.


# Shape of spectra at each break

s_1 = 1.5
s_2 = (1.76 + 0.05*p)
s_3 = (0.8 - 0.03*p)


# Read in and process command line options

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Specify input file for retrieving data', dest='data',type=str,default='None',action='store',required=True)
parser.add_argument('-r', '--raw', help='Plot raw data', dest='raw',default='None',action='store_true',required=False)

args = parser.parse_args()

data_file = args.data
plot_raw_data = args.raw

# Load data

flux = []
freqs = []
error = []
for line in open(data_file):
   lines = line.strip()
   if not line.startswith("#"):
      columns = line.split(',')
      freqs.append(columns[0])
      flux.append(columns[1])
      error.append(columns[2].rstrip('\n'))

flux = np.array(flux).astype(float)
freqs = np.array(freqs).astype(float)
error = np.array(error).astype(float)



# Plot raw data if argument -r passed

if plot_raw_data is True:
   plt.figure()
   plt.scatter(freqs,flux)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()




# Define synchrotron spectrum for model 1 in Granot and Sari

def spectrum(v,F_v,v_a,v_m):
    return F_v * (((v/v_a)**(-s_1*beta_1) + (v/v_a)**(-s_1*beta_2))**(-1./s_1)) * ((1 + (v/v_m)**(s_2*(beta_2-beta_3)))**(-1./s_2))




# Log likelihood function

def lnlike(theta, v, y, yerr):
    F_v,v_a,v_m,lnf = theta
    model = spectrum(v,F_v,v_a,v_m)
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# Log priors

def lnprior(theta):
    F_v,v_a,v_m,lnf = theta
    if (1. < F_v < 55.) and (10**(8.) < v_a < 10**(13.)) and (10**(9.) < v_m < 10**(13.)) and (-3 < lnf < -0.01):
        return 0.0
    return -np.inf

# Log probability

def lnprob(theta, v, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, v, y, yerr)
    
# Define number of dimensions and number of walkers

ndim, nwalkers = 4, 100


# Define initial positions of walkers in phase space

frand = np.random.normal(loc=F_true,size=nwalkers,scale=0.1)
varand = np.random.normal(loc=va_true,size=nwalkers,scale=1.E3)
vmrand = np.random.normal(loc=vm_true,size=nwalkers,scale=1.E3)
yerrand = np.random.normal(loc=-0.7,size=nwalkers,scale=0.1)

pos = np.column_stack((frand,varand,vmrand,yerrand)) 
pos_add_dim = np.expand_dims(pos,axis=0)
final_pos = np.repeat(pos_add_dim, 5, axis=0)

sampler = emcee.PTSampler(5, nwalkers, ndim, lnlike, lnprior, loglargs=[freqs,flux,error])
sams = sampler.run_mcmc(final_pos, 1000)

# Burn off initial steps
samples = sampler.chain[0,:, 500:, :].reshape((-1, ndim))

F_mcmc = np.mean(samples[:,0])
va_mcmc = np.mean(samples[:,1])
vm_mcmc = np.mean(samples[:,2])
lnf_mcmc = np.mean(samples[:,3])


# Print results
print "F_v = %s" % F_mcmc
print "v_a = %s" % va_mcmc
print "v_m = %s" % vm_mcmc

print "Log Likelihood = %s" %lnlike([F_mcmc,va_mcmc,vm_mcmc,lnf_mcmc], freqs, flux, error)

v_range = np.linspace(1E9,350E9,1E4)
plt.scatter(freqs,flux)
plt.plot(v_range,spectrum(v_range,F_mcmc,va_mcmc,vm_mcmc))
plt.xscale('log')
plt.yscale('log')
plt.show()
