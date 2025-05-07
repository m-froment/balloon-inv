# Bayesian inversion of a planet's interior structure using balloon pressure data

Froment, M., Brissaud, Q., Näsholm, S. P. and Schweitzer, J. (2025) _Joint source and subsurface inversion using earthquake-generated infrasound recorded by balloons_

This suite of codes performs the inversion of one component seismograms/pressure data to simultaneously retrieve the seismic source location and the planet's 1D velocity  structure. A Bayesian inversion approach is used, implementing different Markov chain Monte Carlo (McMC) algorithms. The inverted data consists of arrival times of Rayleigh Waves, P and S waves, measured from different types of signals:  
<ul>
  <li>Seismograms (seismic stations)</li>
  <li>Airborne pressure recordings (balloons)</li>
</ul> 
It is also applicable to pressure signals recorded on the ground (microbarometers) and synthetic signals. 

## Requirements 
Some important modules required for running the inversion are: 
<ul>
  <li>Jupyter</li>  
  <li>Obspy</li>
  <li>emcee</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>disba</li>
  <li>f2py</li>
</ul>
As well as mainy others. A complete Python environment is available in <code>inversion_environment.yml</code>. it can be installed with the following command: 
```
conda env create -f inversion_environment.yml
conda activate env_mcmc
```

## Structure of the code 

A walk through a full inversion run (using the Strateole2 balloon data) is presented in the <code>test_inversion_flores_balloons.ipynb</code> notebook. This includes the processing of the balloon data, the extraction of picks, the formating of the data and preparation of the inversion, the inversion run and the final data processing as well as some figure outputs.  

DATE: May 2025. 


## Acknowledgements: 
This study is funded by the AIR project: https://norsarair.github.io/. This code makes use of open-source modules for seismology and McMC inversions, such as [ObsPy](https://docs.obspy.org/) and [emcee](https://emcee.readthedocs.io/en/stable/), and we thank their contributors for providing and maintaining them.  