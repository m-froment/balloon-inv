### OBPSY
import obspy 
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from disba import GroupDispersion
### OBSPY 
import pandas as pd
import numpy as np
### MATPLOTLIB
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, gridspec, ticker
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmcrameri import cm as cmc
import seaborn as sns
###
from skimage.measure import block_reduce
import copy
### SCIPY
from scipy.fft import ifft, fft, fftfreq
from scipy.integrate import trapezoid
from scipy.stats import qmc
from scipy.signal import hilbert, windows
###
import emcee
from ttplanet import ttloc 
from tqdm import tqdm
import sys
import os
import shutil
import time as ptime 
import h5py
import pickle
import random 
#### MULTIPROCESSING PARAMETERS ####
from multiprocessing import Pool, cpu_count
# os.environ["OMP_NUM_THREADS"] = "1"
#### MPIPool PARAMETERS ####
# from schwimmbad import MPIPool
rcParams['savefig.dpi'] = 600


##############################################################################
def vp_from_vs(vs, poisson): 
    # Calculate the P-wave vel. from S wave vel. and Poisson's ratio.
    return vs * np.sqrt((2 * (1 - poisson)) / (1 - 2 * poisson))


##############################################################################
def calculate_poisson_ratio(vp, vs):
    ### Calculate the Poisson's ratio given P-wave and S-wave velocities.
    ### Parameters:
    ###     vp (float): P-wave velocity (any unit)
    ###     vs (float): S-wave velocity (same unit)
    ### Returns:
    ###     float: Poisson's ratio
    
    ### Calculate the square of velocity ratios
    velocity_ratio_squared = (vp / vs) ** 2
    
    #### Calculate Poisson's ratio
    poisson_ratio = (velocity_ratio_squared - 2) / (2 * (velocity_ratio_squared - 1))
    
    return poisson_ratio


##############################################################################
def birch_law(vp): 
    ### Original paper: Birch 1964: Composition of the Earth's mantle 
    ### Gives rho = A(M) + 0.302 Vp, with M mean atomic weight 
    ###       A(21) = 0.77, A(25) = 1.72, A(30) = 2.6
    ### For Mars, Huang (2022) use different parameters: 
    ###       rho = 0.613 + 0.327*Vp
    ###    or rho = (vp+1.87)/0.00305/1e3
    ### Over values can be found using the upper mantle relationships rho(x)
    ### and vp(x) in PREM:
    ### From 25km to 220 km: rho = 2.61 + 0.095*vp 
    ###      220-400 km      rho = 0.779 + 0.310*vp 
    ###      400-600 km      rho = 1.475 + 0.24*vp 
    return 0.77 + 0.302*vp


##############################################################################
def propagation_time_air(z1, model_atm, z0=0):
    ### Time for the propagation of a vertical acoustic waves from z=z0 to z=z1
    ### Station altitude do not vary, so this can be calculated in advance. 
    ### Implemented are: Earth general, Alaska, Flores, Venus 
    ### model_atm is passed to DATA 
    z = model_atm[:,0]  # m
    c = model_atm[:,1]  # m/s

    imin = np.where( abs(z-z0) == np.min(abs(z-z0)))[0][0]
    imax = np.where( abs(z-z1) == np.min(abs(z-z1)))[0][0]

    tair = trapezoid(1/c[imin:imax], x=z[imin:imax])
    return(tair)


######################################################################################################
def create_model(vs, poisson, h_layers, vps=None):
    
    # Velocity model
    # thickness, Vp, Vs, density
    # km, km/s, km/s, g/cm3
    
    velocity_model = []
    
    for cpt_layer, (one_vs, one_poisson) in enumerate(zip(vs, poisson)):

        if vps is None:
            vp = vp_from_vs(one_vs, one_poisson)
        else:
            vp = vps[cpt_layer]
        rho = birch_law(vp)

        if cpt_layer < len(vs)-1:
            one_h = h_layers[cpt_layer]
        else:
            one_h = 1000.
            
        layer = [one_h, vp, one_vs, rho]
        velocity_model.append(layer)
        
    velocity_model = np.array(velocity_model)
    return velocity_model


########################################################################################################
def compute_travel_time_SP(dist, source_depth, velocity_model, re=6371, phase='ps'):
    ### Uses ttplanet, based on laufps 
    ### Structure of the model: z=depth, v0 = [vp,vs], azo = indicator for MOHO depth 
    ### Each interface is doubled here because we want a succession of homogeneous layers 
    ### If there is a MOHO depth, it is the *lowest* of the two depths

    ### Depth
    zdep = np.concatenate(([0.], np.repeat(np.cumsum(velocity_model[:,0]),2) ))
    zdep[-1] = 4000.    ### the deepest depth (some sort of fluid core below which nothing is calculated)

    ### Velocity 
    vpdep = np.concatenate(( np.repeat(velocity_model[:,1],2), [velocity_model[-1,1]] ))
    vsdep = np.concatenate(( np.repeat(velocity_model[:,2],2), [velocity_model[-1,2]] ))

    ### Moho indicator if wanted 
    azdep = np.array(['n   ' for i in range(vpdep.size)])

    jmod = vsdep.shape[0]
    ### One must append zeros so that layer have the same size as expected by Fortran77 
    z = np.zeros(101)
    v0 = np.zeros((2,101))
    azo = ['' for n in range(101)]
    z[:jmod] = zdep
    v0[:,:jmod] = np.stack((vpdep, vsdep))
    azo[:jmod] = azdep

    locgeo = False  # small ray density
    typctl = 1      # no verbosity

    ### CALL TO LAUFPS 
    nphas2,ttc, phs, _, _ = \
                    ttloc(source_depth, dist*360/(2*np.pi*re), z, v0, azo,
                      jmod,re,locgeo,typctl)              

    if nphas2==0 or ttc[0]==0:
        # print("No phase")
        if phase=="ps":
            return(1e10, 1e10)
        else:
            return(1e10)
    else:
        if phase=="ps":
            ttcp = ttc[[aa==b'Pg      ' and ttc[ia]!=0 for ia, aa in enumerate(phs)]]
            ttcs = ttc[[aa==b'Sg      ' and ttc[ia]!=0 for ia, aa in enumerate(phs)]]
            if len(ttcp)==0:
                return(1e10, ttcs)
            elif len(ttcs)==0:
                return(ttcp, 1e10)
            else:
                ### Return P, then S phase 
                return(min(ttcp[:nphas2]),  min(ttcs[:nphas2]) ) 
        elif phase=="p":
            ttcp = ttc[[aa==b'Pg      ' and ttc[ia]!=0 for ia, aa in enumerate(phs)]]
            if len(ttcp)==0:
                return(1e10)
            else:
                ### Return P phase 
                return( min(ttcp[:nphas2]) ) 
        elif phase=="s":
            # print(a)
            ttcs = ttc[[aa==b'Sg      ' and ttc[ia]!=0 for ia, aa in enumerate(phs)]]
            if len(ttcs)==0:
                return(1e10)
            else:
                ### Return S phase 
                return( min(ttcs[:nphas2]) )  


########################################################################################################
def compute_vg_n_layers(periods, velocity_model, max_mode = 1,):
    
    pd_g = GroupDispersion(*velocity_model.T)
    try:
        cgrs = [pd_g(periods, mode=i, wave="rayleigh") for i in range(max_mode)]
    except:
        return None, None

    vg0 = [cgr.velocity for cgr in cgrs]
    periods = [cgr.period for cgr in cgrs]
    
    return periods, vg0


########################################################################################################
def block_mean(array, f, t, fact, log=False):
    res = block_reduce(array, block_size=(1, fact), func=np.mean, cval=np.mean(array))

    if log: 
        ### TODO: DOesn't work 
        t2 = t.copy()
        t = 1/np.linspace(1/t.max(),1/t.min(),  res.shape[1]+1)
        t = 2/(1/t[1:]+1/t[:1])
        fig = plt.figure() 
        plt.plot(1/t, t)
        plt.plot(1/t2,t2)
        plt.show()
    else:
        t = np.linspace(t[0], t[-1], res.shape[1]+1)
        t = (t[1:]+t[:-1])/2

    return f, t, res


##############################################################################
def mean_before(signal, twin, dt):
    nwin = int(twin//dt) 
    before_mean = np.zeros(signal.shape)

    ### Using a mean 
    # cumsum = np.cumsum(np.insert(signal, 0, 0))
    # before_mean[nwin-1:] = (cumsum[nwin:] - cumsum[:-nwin]) / nwin
    # for j in range(nwin):
    #     before_mean[j] = np.sum(signal[:j+1])/(j+1)

    ### Using a median 
    signal_series = pd.Series(signal)
    before_mean = signal_series.rolling(window=nwin, center=True).median()

    return(before_mean)


##############################################################################
def scientific_10(x, pos):
    ### Function to format ticks with a scientific notation
	if abs(x)==0:
		return r"${: 1.0f}$".format(x)
	else :
		exponent = np.floor(np.log10(abs(x)))
		coeff = x/10**exponent
		if coeff ==1 or coeff==-1:
			return r"$10^{{ {:.0f} }}$".format(exponent)
		else :
			return r"${: 2.0f} \times 10^{{ {:.0f} }}$".format(coeff,exponent)


##############################################################################
def pow_10(x, pos):
    ### Function to format ticks with a power of 10 notation
	if x==-np.inf:
		return r"${: 1.0f}$".format(0)
	else :
		exponent = np.floor(x)
		coeff = 10**x/10**exponent
		if coeff ==1 or coeff==-1:
			return r"$10^{{ {:.0f} }}$".format(exponent)
		else :
			return r"${: 3.2g}$".format(10**x) 


##############################################################################################################
def convert_model_to_theta(velocity_model, n_layers=None):
    vp = velocity_model[:,1]
    vs = velocity_model[:,2]
    h_layers = velocity_model[:-1,0]
    poisson = calculate_poisson_ratio(vp, vs)
    return np.r_[vs,poisson,h_layers]


########################################################################################################
def get_model(x, id_order):
    ### Get variables from a model "theta" or "x"
    t0 = x[id_order[0] : id_order[1]][0]
    source_lat =  x[id_order[1] : id_order[2]][0]
    source_lon = x[id_order[2] : id_order[3]][0]
    source_depth = x[id_order[3] : id_order[4]][0]
    vs = x[id_order[4] : id_order[5]]
    poisson_layers = x[id_order[5] : id_order[6]]
    h_layers = x[id_order[6] : ]
    return(t0, source_lat, source_lon, source_depth, vs, poisson_layers, h_layers)


########################################################################################################
def build_model_from_thicknesses(velocity_model, depth_interpolation, idx_requested):
    
    depths = np.cumsum(velocity_model[:,0])
    if idx_requested == 1 or idx_requested==2:
        vss = velocity_model[:,idx_requested]
    elif idx_requested == 3: 
        vss = (velocity_model[:,1]**2 - 2*velocity_model[:,2]**2) / \
                (2*(velocity_model[:,1]**2 - velocity_model[:,2]**2))
    
    vss_new, depths_new = [vss[0]], [0.]
    for idepth, (depth, vs) in enumerate(zip(depths[:-1], vss[:-1])):
        depths_new.append(depth-0.01)
        vss_new.append(vs)
        depths_new.append(depth)
        vss_new.append(vss[idepth+1])
        
    depths_new.append(depths[-1])
    vss_new.append(vss[-1])

    ### Testing 
    ### The difference with this method is that it doesn't have a doubling at each interface
    # fig, ax = plt.subplots()
    # print(2)
    # ax.plot(np.interp(depth_interpolation, depths_new, vss_new), depth_interpolation, 'o-')
    # for i in range(depths.size):
    #     ax.plot([vss[i],vss[i]],[depths[i]-velocity_model[i,0],depths[i]], ls='', marker='s', c='k')
    # ax.plot([vss[-1],vss[-1]],[depths[-1],np.max(depth_interpolation)], ls='', marker='s', c='k')
    # ax.set_ylim(np.max(depth_interpolation)+1,-1)
    # ax.grid()
    # print(velocity_model)
    # quit()

    return np.interp(depth_interpolation, depths_new, vss_new)


########################################################################################################
def build_model_from_thicknesses_sharp(vs, vp, ps, h_layer, depth_interpolation, dh, N_layer, N_intp, test=False, do_out_depth= False):
    ### Slightly slower but has doubling at each interface
    d_layer= np.cumsum(h_layer)

    depth_val = [] 
    vs_val = []
    vp_val = []
    ps_val = []

    for il in range(N_layer) :
        ### If it is the shallowest layer:
        if il ==0 :
            wi = int(d_layer[il]//dh)
            if d_layer[il]%dh > dh/2:
                wi+=1
            whvs = [0,wi]
        ### If it is an intermediate layer:
        elif il < N_layer :
            wi = int(d_layer[il]//dh)
            if d_layer[il]%dh > dh/2:
                wi+=1
            whvs = [whvs[-1], wi]
            # print(whvs, d_layer[il], depth_interpolation[whvs[1]], dh)
        depth_val+=list(depth_interpolation[whvs[0]:whvs[1]+1])
        vs_val+=[vs[il] for k in range(whvs[1]-whvs[0]+1)]
        vp_val+=[vp[il] for k in range(whvs[1]-whvs[0]+1)]
        ps_val+=[ps[il] for k in range(whvs[1]-whvs[0]+1)]
    # If it is the bottom halfspace
    depth_val+=list(depth_interpolation[whvs[1]:])
    vs_val+=[vs[il+1] for k in range(N_intp-whvs[1])]
    vp_val+=[vp[il+1] for k in range(N_intp-whvs[1])]
    ps_val+=[ps[il+1] for k in range(N_intp-whvs[1])]

    # if test:
    #     # print(len(depth_val), len(depth_interpolation))

    #     fig, ax = plt.subplots()
    #     ax.plot(vs_val, depth_val, 'o-')
    #     ax.plot(vp_val, depth_val, 'o-')
    #     ax.plot(ps_val, depth_val, 's-')
    #     for i in range(d_layer.size):
    #         ax.plot([vs[i],vs[i]],[d_layer[i]-h_layer[i],d_layer[i]], ls='', marker='s', c='k')
    #         ax.plot([vp[i],vp[i]],[d_layer[i]-h_layer[i],d_layer[i]], ls='', marker='^', c='k')
    #         ax.plot([ps[i],ps[i]],[d_layer[i]-h_layer[i],d_layer[i]], ls='', marker='*', c='k')
    #     ax.plot([vs[-1],vs[-1]],[d_layer[-1],np.max(depth_interpolation)], ls='', marker='s', c='k')
    #     ax.plot([vp[-1],vp[-1]],[d_layer[-1],np.max(depth_interpolation)], ls='', marker='^', c='k')
    #     ax.plot([ps[-1],ps[-1]],[d_layer[-1],np.max(depth_interpolation)], ls='', marker='*', c='k')
    #     ax.set_ylim(np.max(depth_interpolation)+1,-1)
    #     ax.grid()
    #     print(vs, vp, ps, d_layer)
    #     #quit()   

    if not do_out_depth:
        return(vs_val, vp_val, ps_val)
    else:
        return(vs_val, vp_val, ps_val, depth_val)


##########################################################################################################  
def _interpolate_one_model_(i, flat_samples_i, blobs_i, is_inverted, wr_order, depth_interpolation, int_method, dh, N_layer, N_intp):
    """ Helper function to process a single model for interpolation """
    if blobs_i is None:
        vs = flat_samples_i[wr_order[3]]
        poisson = flat_samples_i[wr_order[4]]
        h_layer = flat_samples_i[wr_order[5]]
    else:
        vs, poisson, h_layer = get_model_from_blob(flat_samples_i, blobs_i, is_inverted, wr_order)
    
    if int_method == "1":
        velocity_model = create_model(vs, poisson, h_layer)
        vs_interp = build_model_from_thicknesses(velocity_model, depth_interpolation, 2)
        vp_interp = build_model_from_thicknesses(velocity_model, depth_interpolation, 1)
        ps_interp = build_model_from_thicknesses(velocity_model, depth_interpolation, 3)
    elif int_method == "2":
        vp = vp_from_vs(vs, poisson)
        vs_interp, vp_interp, ps_interp = build_model_from_thicknesses_sharp(
            vs, vp, poisson, h_layer, depth_interpolation, dh, N_layer, N_intp
        )
    
    return i, vs_interp, vp_interp, ps_interp

    

########################################################################################################
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


########################################################################################################
def autocorr_func_1d(chain):
    ### Calculate autocorrelation function for 1 chain (emcee)
    ### Following https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    chain = np.atleast_1d(chain)
    if len(chain.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(chain))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(chain - np.mean(chain), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(chain)].real
    acf /= 4 * n
    # normalize
    if acf[0] ==0 :
        return acf
    else:
        acf /= acf[0]
        return acf


########################################################################################################
def auto_window(taus, c):
    ### Automated windowing procedure following Sokal (1989) (emcee)
    ### Following https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

########################################################################################################
def autocorr_new(chain, c=5.0):
    ### Calculate autocorrelation function for 1 chain (emcee)
    ### Following https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    f = autocorr_func_1d(chain)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


########################################################################################################
def plot_contour(ax, Nparam, var1, var2, range_hist, sigmas=np.array([1,2,3]) ):
    ### Function for plotting 2D 1-sigma, 2-sigma, 3-sigma contours in 2D distributions  
     
    ### Amplitude of N-sigma contours for a 2D distribution. 
    levels = 1.0 - np.exp(- sigmas**2 / 2.0)

    ### Calculate histogram 
    H, X, Y = np.histogram2d(var1,var2, bins=20, range = range_hist )
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    ### Calculate levels from histograms 
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        print("Too few points to create valid contours")
        return()

    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]

    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )
    ax.contour(X2, Y2, H2.T, V, colors='k', linewidths=0.5*Nparam/10)
    return()



########################################################################################################
def get_model_from_inverted(theta, wr_order, prior_lims, is_inverted):
    ### FUNCTION TO GET A FULL MODEL IN CASE AN INVERSION HAS 
    ### INVERTED AND NON-INVERTED VARIABLES 
    ### THE ORDER MUST NEVER CHANGE !

    ### TODO: Update. It is too complicated 

    cvar = 0 
    blob = []
    if is_inverted[0]: 
        t0 = theta[wr_order[0]]
    else: 
        ### np random returns lowest value if both values are equal. 
        t0 = np.random.uniform(prior_lims[cvar][0], prior_lims[cvar][1])
        blob.append(t0)
    cvar+=1
    if is_inverted[1][0]:
        source_lat = theta[wr_order[1][0]]
    else:
        source_lat  = np.random.uniform(prior_lims[cvar][0], prior_lims[cvar][1])
        blob.append(source_lat)
    cvar+=1
    if is_inverted[1][1]:
        source_lon = theta[wr_order[1][1]]
    else:
        source_lon  = np.random.uniform(prior_lims[cvar][0], prior_lims[cvar][1])
        blob.append(source_lon)
    cvar+=1
    if is_inverted[2]:
        source_depth = theta[wr_order[2]]
    else:
        source_depth  = np.random.uniform(prior_lims[cvar][0], prior_lims[cvar][1])
        blob.append(source_depth)
    cvar+=1
    ### More complex: Check if and where vs is inverted 
    vs = np.zeros(len(is_inverted[3]))
    if np.any(is_inverted[3]):
        vs[is_inverted[3]] = theta[wr_order[3]]
    if not np.all(is_inverted[3]):
        for i, iv in enumerate(is_inverted[3]): 
            if not iv: 
                vs[i] = np.random.uniform(prior_lims[cvar+i][0], prior_lims[cvar+i][1])
                blob.append(vs[i])
    cvar+=vs.size
    ### Check if and where Poisson is inverted 
    poisson_layers = np.zeros(len(is_inverted[4]))
    if np.any(is_inverted[4]):
        poisson_layers[is_inverted[4]] = theta[wr_order[4]]
    if not np.all(is_inverted[4]):
        for i, iv in enumerate(is_inverted[4]): 
            if not iv: 
                poisson_layers[i] = np.random.uniform(prior_lims[cvar+i][0], prior_lims[cvar+i][1])
                blob.append(poisson_layers[i])
    cvar+=poisson_layers.size
    ### Check if and where Layer thickness is inverted 
    h_layers = np.zeros(len(is_inverted[5]))
    if np.any(is_inverted[5]):
        h_layers[is_inverted[5]] = theta[wr_order[5]]
    if not np.all(is_inverted[5]):
        for i, iv in enumerate(is_inverted[5]): 
            if not iv: 
                h_layers[i] = np.random.uniform(prior_lims[cvar+i][0], prior_lims[cvar+i][1])
                blob.append(h_layers[i])

    return(t0, source_lat, source_lon, source_depth, vs, poisson_layers, h_layers, blob)


########################################################################################################
def get_model_from_blob(inv_param, noinv_param, is_inverted, wr_order):
    ### FUNCTION TO POPULATE VS; PS; HL GIVEN INVERTED PARAMETERS AND FIXED PARAMETERS 
    ### Note: Based on get_model_from_inverted()
    ### TODO: Delete. Not a good method 

    cvar = 0 
    vs = np.zeros(len(is_inverted[3]))
    if np.any(is_inverted[3]):
        vs[is_inverted[3]] = inv_param[wr_order[3]]
    if not np.all(is_inverted[3]):
        for i, iv in enumerate(is_inverted[3]): 
            if not iv: 
                vs[i] = noinv_param["fixed_var_{:d}".format(cvar)]
                cvar+=1
    poisson = np.zeros(len(is_inverted[4]))
    if np.any(is_inverted[4]):
        poisson[is_inverted[4]] = inv_param[wr_order[4]]
    if not np.all(is_inverted[4]):
        for i, iv in enumerate(is_inverted[4]): 
            if not iv: 
                poisson[i] = noinv_param["fixed_var_{:d}".format(cvar)]
                cvar+=1
    h_layer = np.zeros(len(is_inverted[5]))
    if np.any(is_inverted[5]):
        h_layer[is_inverted[5]] = inv_param[wr_order[5]]
    if not np.all(is_inverted[5]):
        for i, iv in enumerate(is_inverted[5]): 
            if not iv: 
                h_layer[i] = noinv_param["fixed_var_{:d}".format(cvar)]
                cvar+=1
    return(vs, poisson, h_layer)
############################################################################ 


########################################################################################################
def plot_subsurface(ax, val_x, val_y, Ndepth, vmin, vmax, depth_max, my_cmap, bgcol="grey", dmin= 6, aspect_ratio=None, hexplot=False): 

    NPT = val_x.size
    NTOT = NPT/Ndepth 

    ### Define histogram dimensions
    if hexplot:
        nhei = 50
        nwid = int((nhei+15*nhei/50)*ax.bbox.width/ax.bbox.height)  # Empirical 
    else:
        nhei,nwid = 100,50
    if aspect_ratio is not None:
        nwid = int(nwid*aspect_ratio)

    if hexplot:
        ### Smaller hexagons always have less points so it skews the plot
        ### Try to guesstimate the max number of poimts per hexagon. 
        vmaxn = NTOT*(Ndepth/nhei)
        vminn = 0
        ### Potential value for hexbin plot 
        valmin = 0
        valmax = 1e-1
    else:
        ### Min val for density plot: if a bin contains only one count 
        valmin = 1 / (NPT*(vmax -vmin)/nwid*(depth_max-0)/nhei)
        ### Max val for density plot: if 3 times the mean number per bin 
        valmax = dmin / (nhei*nwid * (vmax -vmin)/nwid*(depth_max-0)/nhei)

    ax.set_facecolor(bgcol)
    if hexplot:
        ### OPTION 1: HEXAGON BINS (TODO: Not tested in a while)
        poly = ax.hexbin(val_x, val_y, 
                        C=np.ones((val_x.size))/vmaxn,reduce_C_function=np.sum, 
                        gridsize=(nwid, nhei),cmap=my_cmap, extent=[vmin, vmax, 0, depth_max],
                        rasterized = True, linewidths=0.3, mincnt=1, vmin=valmin, vmax=valmax )  
        hvac = poly.get_array()
    else:
        ### OPTION 2: 2D BINS 
        _, _,_,hvac = ax.hist2d(val_x, val_y,  
                        bins=(nwid, nhei),cmap=my_cmap, range=[[vmin, vmax], [0, depth_max]],
                        rasterized = True, cmin=valmin, density=True,
                        vmin=valmin, vmax=valmax)
    return(hvac, valmin, valmax)



########################################################################################################
def log_prior(x, prior_inv, wr_order):
    wrong = any((prior_inv[i][0]>mm) | (mm>prior_inv[i][1]) for i, mm in enumerate(x))
    if wrong :
        return(-np.inf)
    else:
        picked_vs = x[wr_order[3]]
        ### Reject models with a LVZ in the first three layers (crust) in VS 
        if np.any(np.diff(picked_vs[:2])<0):
            return(-np.inf)
        ### Else reject models with very strong LVZ (-1 km/s) in VS 
        if picked_vs.size:
            if np.any(np.diff(picked_vs[:-1])<-1):
                return(-np.inf)
        ### In vp 
        picked_ps = x[wr_order[4]]
        if picked_ps.size:
            vp = vp_from_vs(picked_vs, picked_ps)
            ### Reject models with a LVZ in the first three layers (crust) in VP
            if np.any(np.diff(vp[:2])<0):
                return(-np.inf)
            ### Else reject models with very strong LVZ (-1 km/s) in VP 
            if np.any(np.diff(vp[:-1])<-1):
                return(-np.inf)
            ### Next check that there isn't any vp larger than 12 
            if np.any(vp>12):
                return(-np.inf)
        return(0)


###################################################################################################################
def log_likelihood(theta, wr_order, prior_lims, is_inverted, data_vector, std_vector, sta_lats, sta_lons, tprops_air, periods_RWs):

    ### Retrieve model parameters 
    t0, source_lat, source_lon, source_depth, vs, poisson_layers, h_layers, blob = get_model_from_inverted(theta, wr_order, prior_lims, is_inverted)

    velocity_model = create_model(vs, poisson_layers, h_layers)
    
    ### Initialize total log-likelihood 
    score = 0.
    # Load station location data and propagation time in air (zero for surface station)
    for ii, (sta_lat, sta_lon, tair) in enumerate(zip(sta_lats, sta_lons, tprops_air)):
        
        ### Station-Event great circle distance in km 
        sta_dist = gps2dist_azimuth(sta_lat, sta_lon, source_lat, source_lon)[0]/1e3 


        ### If inverting using Rayleigh Waves:     
        if not np.any(data_vector[ii]["RW_arrival"] == None) : 
            periods_RW = periods_RWs[ii]
            _, predicted_vg0_RW = compute_vg_n_layers(periods_RW, velocity_model, max_mode = 1,)
            
            if predicted_vg0_RW is None or predicted_vg0_RW[0].size < data_vector[ii]["RW_arrival"].size:
                ### rejecting: no vg found
                return -np.inf, blob 

            ### Arrival time of Rayleigh Wave: 
            trw = t0 + (sta_dist)/predicted_vg0_RW[0] + tair 
            ### Calculate log of L2 misfit 
            score_RW = -1/2 * np.sum( ( (trw - data_vector[ii]["RW_arrival"]) /std_vector[ii]["RW_error"] )**2 ) -1/2*np.sum(np.log(2*np.pi*std_vector[ii]["RW_error"] **2))
        else: 
            score_RW = 0 

        ### If inverting using both P and S waves: 
        score_S = 0 
        score_P = 0

        ### If inverting using both P and S arrivals 
        if data_vector[ii]["S_arrival"] != None and data_vector[ii]["P_arrival"] != None:
            dtP, dtS = compute_travel_time_SP(sta_dist, source_depth, velocity_model.copy(), phase="ps")
            if dtP >1e9 or dtS>1e9:
                ### rejecting: no P found
                return -np.inf, blob 
            else:
                score_S = -1/2 * ( (t0 + dtS + tair -data_vector[ii]["S_arrival"])/std_vector[ii]["S_error"] )**2 -1/2*np.log(2*np.pi*std_vector[ii]["S_error"] **2)
                score_P = -1/2 * ( (t0 + dtP + tair -data_vector[ii]["P_arrival"])/std_vector[ii]["P_error"] )**2  -1/2*np.log(2*np.pi*std_vector[ii]["P_error"] **2)
                    

        ### If inverting only using S waves: 
        elif data_vector[ii]["S_arrival"] != None:
            
            ### Compute S wave travel time 
            dtS = compute_travel_time_SP(sta_dist, source_depth, velocity_model.copy(), phase="s")

            if dtS >1e9:
                ### rejecting: no S found
                return -np.inf, blob 
            else:
                ### Calculate log of L2 misfit 
                score_S = -1/2 * ( (t0 + dtS + tair -data_vector[ii]["S_arrival"])/std_vector[ii]["S_error"] )**2 -1/2*np.log(2*np.pi*std_vector[ii]["S_error"] **2)
                #print("tS : ", dtS)

        ### If inverting only using P waves: 
        elif data_vector[ii]["P_arrival"] != None:
            
            ### Compute P wave travel time 
            dtP = compute_travel_time_SP(sta_dist, source_depth, velocity_model.copy(), phase="p")

            if dtP >1e9:
                ### rejecting: no P found
                return -np.inf, blob 
            else:
                ### Calculate log of L2 misfit 
                score_P = -1/2 * ( (t0 + dtP + tair -data_vector[ii]["P_arrival"])/std_vector[ii]["P_error"] )**2  -1/2*np.log(2*np.pi*std_vector[ii]["P_error"] **2)
                #print("tP : ", dtP)
 
        score += score_S + score_RW + score_P
        
    ### We also return the blob containing all parameters that are fixed or not inverted
    return score, blob 


########################################################################################################
def log_probability(x, prior_lims, prior_inv, is_inverted, wr_order, args_MCMC):
    l_novinv = len(prior_lims)-len(prior_inv)
    lp = log_prior(x, prior_inv, wr_order)
    ### ABSOLUTELY **OBSCENE** BUT CAN'T FIND A WAY TO RETURN BLOBS AS ARRAY IN EMCEE
    ### TODO: Find a different way to do this. Or delete it fully, blobs are not needed. 
    if not np.isfinite(lp):
        if l_novinv == 0:
            return -np.inf 
        elif l_novinv == 1: 
            return -np.inf, None 
        elif l_novinv == 2: 
            return -np.inf, None , None 
        elif l_novinv==3:
            return -np.inf, None , None , None 
        elif l_novinv==4:
            return -np.inf, None , None , None , None
        elif l_novinv==5:
            return -np.inf, None , None , None , None , None  
        elif l_novinv==6:
            return -np.inf, None , None , None , None , None , None
        elif l_novinv==7:
            return -np.inf, None , None , None , None , None , None , None
        elif l_novinv==8:
            return -np.inf, None , None , None , None , None , None , None , None
        else:
            raise ValueError("Connot define blobs for more than 8 variables")
    ll, blob = log_likelihood(x, wr_order, prior_lims, is_inverted, *args_MCMC)
    if l_novinv == 0:
        return lp + ll
    elif l_novinv == 1: 
        return lp + ll, blob[0] 
    elif l_novinv == 2: 
        return lp + ll, blob[0] , blob[1] 
    elif l_novinv==3:
        return lp + ll, blob[0] , blob[1] , blob[2] 
    elif l_novinv==4:
        return lp + ll, blob[0] , blob[1] , blob[2] , blob[3]
    elif l_novinv==5:
        return lp + ll, blob[0] , blob[1] , blob[2] , blob[3] , blob[4]  
    elif l_novinv==6:
        return lp + ll, blob[0] , blob[1] , blob[2] , blob[3] , blob[4] , blob[5]
    elif l_novinv==7:
        return lp + ll, blob[0] , blob[1] , blob[2] , blob[3] , blob[4] , blob[5] , blob[6]
    elif l_novinv==8:
        return lp + ll, blob[0] , blob[1] , blob[2] , blob[3] , blob[4] , blob[5] , blob[6] , blob[7]
    else:
        raise ValueError("Connot define blobs for more than 8 variables")


########################################################################################################
def generate_initial_model(n_dim, n_temp, n_walk, start_lims, wr_order, prior_lims, is_inverted):
    ### Sample initial model parameters using Latin Hypercube sampling.
    ### Sample over n_dim parameters 
    sampler = qmc.LatinHypercube(d=n_dim)
    sample = sampler.random(n=n_temp*n_walk)
    ### Rescale according to lower and upper prior bounds
    l_bounds = [s[0] for s in start_lims]
    u_bounds = [s[1] for s in start_lims]
    mod0 = qmc.scale(sample, l_bounds, u_bounds)

    ### Ensure here that vs and vp are monotonous, in a smart way 
    wr_vs = wr_order[3]
    wr_ps = wr_order[4]
    for iw in range(n_walk*n_temp):
        _, _, _, _, vs, poisson_layers, h_layers, blob = get_model_from_inverted(mod0[iw,:], wr_order, prior_lims, is_inverted)
        vmod0 = create_model(vs, poisson_layers, h_layers)
        if np.any(np.diff(vmod0[:,2])<0):
            while (np.any(np.diff(vmod0[:,2])<0)):
                alvz = np.where(np.diff(vmod0[:,2])<0)[0]
                ### Remove LVZ 
                for ia in alvz:
                    vmod0[ia+1,2] = vmod0[ia,2]
            mod0[iw,wr_vs] = vmod0[is_inverted[3],2]
        if np.any(np.diff(vmod0[:,1])<0):
            while (np.any(np.diff(vmod0[:,1])<0)):
                alvz = np.where(np.diff(vmod0[:,1])<0)[0]
                ### Remove LVZ 
                for ia in alvz:
                    vmod0[ia+1,1] = vmod0[ia,1]
        ps = calculate_poisson_ratio(vmod0[:,1], vmod0[:,2])
        ps [np.where(ps>=0.5)] = 0.45
        mod0[iw,wr_ps] = ps[is_inverted[4]]
        ### NOTE: With this method, there is still a possibility that both 
        ### vs and ps are outside prior bounds 
        ### Here at least we avoid very ill-posed models

    ### If needed, reshape the model set
    mod0 = np.reshape(mod0, (n_temp,n_walk,n_dim))
    return(mod0)


########################################################################################################
def move_model(x,proposal_cov):
    ### For ptmcmc
    ### Applies a gaussian step from previous position. 
    x_new = np.random.multivariate_normal(x, proposal_cov)
    return(x_new)



########################################################################################################
### CLASS FOR PREPARING THE DATA FOR THE INVERSION 
########################################################################################################
class MCMC_data():

    ####################################################################################################
    def __init__(self, traces, stations_info, event_info, true_velocity_model, select,
                    target_periods, t_ref=0, periods=np.linspace(1., 20., 100),
                    plot=False, data_dir=None, model_atm=None, 
                    initialize =True):

        # Local dictionary of all local variables (arguments)
        locals_dict = locals()
        # Iterate through each item in the dictionary
        for name, value in locals_dict.items():
            if name != "self":
                # Set each as an attribute of the instance
                if name == "stations_info":
                    self.sta_lons, self.sta_lats, self.sta_alts, self.dist_km, self.dist_deg =  value 
                elif name == "event_info": 
                    self.ev_lon, self.ev_lat, self.depth, t_event = value
                    self.t_event = UTCDateTime(t_event)
                elif name =="t_ref":
                    self.t_ref = UTCDateTime(t_ref)
                else:
                    setattr(self, name, value)

        ### Variable for interactive picking figure: 
        self.count_ev = 0
        ### Default: read traces in .mseed format
        self.key_traces =1

        ########################################################################################
        ### Build inversion data 
        if self.initialize:
            print("Building data to invert")
            ### Calculate travel times in air
            self.tprops_air = np.zeros(len(self.sta_alts))
            if self.model_atm is not None: 
                for ii, zalt in enumerate(self.sta_alts):
                    self.tprops_air[ii] = propagation_time_air(zalt, self.model_atm)

            self.truth_location, self.truth_velocity, self.std_Ss, self.arrival_Ss, self.std_RWs, self.arrival_RWs, self.std_Ps, \
                    self.arrival_Ps, self.periods_RWs, self.picked_vgs, self.data_vector, self.std_vector = \
                    self.build_truth()

            print("Saving MCMC Files")
            ### Save all the useful data: 
            files_to_save = ["mcmcdata_truth_location", "mcmcdata_truth_velocity", "mcmcdata_sigmaS", "mcmcdata_arrivalS", "mcmcdata_sigmaRW", 
                            "mcmcdata_arrivalRW", "mcmcdata_sigmaP", "mcmcdata_arrivalP", "mcmcdata_periodRW",
                            "mcmcdata_groupvelRW", "mcmcdata_datavector", "mcmcdata_sigmavector", "mcmcdata_tairprop"]
            data_to_save = [self.truth_location, self.truth_velocity, self.std_Ss, self.arrival_Ss, self.std_RWs, self.arrival_RWs, self.std_Ps, self.arrival_Ps, 
                            self.periods_RWs, self.picked_vgs, self.data_vector, self.std_vector, self.tprops_air]
            for fi, ds in enumerate(files_to_save):
                with open(self.data_dir + ds + "_pik", "wb") as fp:
                    pickle.dump(data_to_save[fi], fp)
        
        ########################################################################################
        ### Recover the saved inversion data: 
        else:
            print("Loading data to invert")
            files_to_open = ["mcmcdata_truth_location", "mcmcdata_truth_velocity", "mcmcdata_sigmaS", "mcmcdata_arrivalS", "mcmcdata_sigmaRW", 
                            "mcmcdata_arrivalRW", "mcmcdata_sigmaP", "mcmcdata_arrivalP", "mcmcdata_periodRW",
                            "mcmcdata_groupvelRW", "mcmcdata_datavector", "mcmcdata_sigmavector", "mcmcdata_tairprop"]
            data_to_open = ["truth_location", "truth_velocity", "std_Ss", "arrival_Ss", "std_RWs", "arrival_RWs", "std_Ps", "arrival_Ps", 
                            "periods_RWs", "picked_vgs", "data_vector", "std_vector", "tprops_air"]
            for fi, ds in enumerate(files_to_open):
                with open(self.data_dir + ds + "_pik", "rb") as fp:
                    dd = pickle.load(fp)
                    setattr(self, data_to_open[fi], dd) 


    #####################################################################################################
    def build_truth(self):
        '''
        Code does a loop on each balloon/seismic trace
            1. Calculate various FTAN 
            2. Extract S anf P arrival interactively
            3. Extract RW and STD interactively
            4. Optional: Plot the picks
            5. FINAL: Build the 'data' vectors for inversion  
            6. FINAL: Build a "Truth" vector that will be used to evaluate inversion results 
        '''

        ### Initialise Ss vector, RWs vector 
        std_Ss, arrival_Ss, std_Ps, arrival_Ps, std_RWs, arrival_RWs, periods_RWs, picked_vgs, picked_fgs, picked_deltavgs = [], [], [], [], [], [], [], [], [], []
        ### Initialise data vector and std vector 
        data_vector, std_vector = [], []

        ### Loop on multiple signals, if multiple signals are available 
        for ii in range(len(self.traces)):

            ### Load waveform and source-receiver distance, source depth (IF KNOWN). 
            if self.key_traces == 0: ### Synthetic waveforms 
                waveform = self.traces[ii]
            else:  ### Obspy waveforms 
                waveform = self.traces[ii]
            dist = self.dist_km[ii]*1e3    ### in meters
            ddeg = self.dist_deg[ii]       ### in degrees 


            """ STEPS 1. 2. 3.: proceed to the picking """
            ### Extract arrival times and dispersion curves, from signal 
            a_ftan, p_FTAN, t_ftan, picked_t, picked_vg, picked_fg, delta_vg, periods_RW_modes, vels_RW_modes, \
                    arrival_time_P, arrival_time_S, std_RW, std_P, std_S = self.FTAN_and_arrival_times(waveform, dist, ii)

            """ STEP 4. Optional: plot extracted picks"""
            ### Plot extracted picks and dispersion curves together with signal
            if self.plot:
                self.plot_waveform( waveform, a_ftan, 
                              ii, dist, p_FTAN, t_ftan, periods_RW_modes, vels_RW_modes, 
                              picked_fg, picked_t, std_RW, std_P, std_S, 
                              arrival_time_P, arrival_time_S,  plot_extracted=True)
                plt.show(block=True)
            
            """FINAL STEP: 5. Construct the 'data' vector for the inversion"""
            picked_fgs.append(picked_fg)
            periods_RWs.append(1/picked_fg)
            arrival_RWs.append( picked_t )
            std_RWs.append(std_RW)      
            ### Sanity check: the group velocity if the source-receiver distance is known
            picked_vgs.append(picked_vg)               
            arrival_Ss.append( arrival_time_S)        
            arrival_Ps.append( arrival_time_P)
            std_Ss.append(std_S)
            std_Ps.append(std_P)
            
            ### Construct data and std vectors 
            ### They are stored in a dictionnary 
            data_vector.append( {"S_arrival":arrival_Ss[ii], "RW_arrival":arrival_RWs[ii], "P_arrival":arrival_Ps[ii] })
            std_vector.append( {"S_error":std_Ss[ii], "RW_error":std_RWs[ii], "P_error":std_Ps[ii]})

        '''STEP 6. FINAL: Build a "Truth" vector'''
        ### If the event time is known, then t_ref = t_event and ts = 0
        ### The order of the source vector follows the one used in the inversion:
        ### ts , lats, lons , hs 
        truth_location = np.array([self.t_event-self.t_ref, self.ev_lat, self.ev_lon, self.depth/1e3] )
        truth_velocity = self.true_velocity_model 

        return truth_location, truth_velocity, std_Ss, arrival_Ss, std_RWs, arrival_RWs, std_Ps, arrival_Ps, periods_RWs, picked_vgs, data_vector, std_vector


    #######################################################################################################
    def plot_waveform(self,   waveform, a_FTAN, 
                              ievent, dist, p_FTAN, t_FTAN, periods_RW_modes, vels_RW_modes, 
                              picked_fg, picked_t, std_RW, std_P, std_S, 
                              arrival_time_P, arrival_time_S,  plot_extracted=True, fontsize=12.,
                              downsample = False, max_t_samples=2000
    ):

        ####################################################
        if self.key_traces == 1:
            otime= waveform.times(reftime=self.t_ref)
            signal = waveform.data[otime>0]
            time = otime[otime>0]
        else:
            time = waveform.get_xdata()
            signal = waveform.get_ydata()[time>0]
            time = time[time>0]
        dt= np.diff(time)[0]
        t0 = 0 
        tair = self.tprops_air[ievent]
        
        #####################################################
        fig = plt.figure(figsize=(8,7))
        grid = fig.add_gridspec(3, 1)
        ax_map = fig.add_subplot(grid[:2,0])
        colmap = cmc.lipari
        #####################################################

        ####################################################
        ### Normalize by overall max 
        # a_FTAN/=np.max(a_FTAN)
        ### Normalize by max at each frequency
        # a_FTAN/=np.max(a_FTAN, axis=1, keepdims=True)
        ### Log 
        a_FTAN = 10*np.log10(abs(a_FTAN))
        f_FTAN = 1/p_FTAN
        ### Remove later parts of signal 
        if t_FTAN[-1]> 2*dist*1e-3/2.5:
            it = np.where(t_FTAN>dist*1e-3/1.)[0][0]
        else:
            it = None
        t_FTAN = t_FTAN[:it]
        a_FTAN = a_FTAN[:,:it]

        ### Downsampling for faster plotting
        if downsample or a_FTAN.shape[1]>max_t_samples:
            print("Array downsampled")
            ### time interpolation to have less than max_t_samples. 
            downsample_fact = max(10,int(a_FTAN.shape[1]//max_t_samples))
            f_FTAN, t_FTAN, a_FTAN = block_mean(a_FTAN, f_FTAN, t_FTAN, downsample_fact)

        ### Find max around the center of the spectrogram
        wrf = np.where((f_FTAN>1e-1) & (f_FTAN<1e0))
        vmin = np.mean(a_FTAN[wrf]) - 3*np.std(a_FTAN[wrf])
        vmax = a_FTAN[wrf].max()
            
        ### Plot with pcolormesh
        ax_map.pcolormesh(t_FTAN, f_FTAN, a_FTAN, cmap=colmap, rasterized=True, vmin=vmin, vmax=vmax)#, shading="nearest")
        

        ####################################################
        ### Plot extracted RW curve 
        if plot_extracted: 
            ax_map.plot(picked_t, picked_fg, color='orange', label='Extracted c$_g$')
            ax_map.fill_betweenx(picked_fg, 
                                    picked_t-std_RW, 
                                    picked_t+std_RW, 
                                    color="orange", linewidth=0.8, alpha=0.2)
            ax_map.errorbar(picked_t, picked_fg,  
                            xerr = std_RW , 
                            color="orange", marker="+", elinewidth=1, capsize=2)

        ####################################################
        ### Plot known modes 
        nm=len(vels_RW_modes) 
        cmap = sns.color_palette("crest", as_cmap=True)#cm.get_cmap('Spectral')
        cols = [cmap(0+(i/nm*0.7) ) for i in range(nm)]
        lss = ['-', '-', '-']
        if nm>3:
            lss += [':' for i in range(3,nm)]
        for imode, (periods_RW, vels_RW) in enumerate(zip(periods_RW_modes, vels_RW_modes)):
            label = dict(label=f'Theor. mode {imode}')
            ax_map.plot(t0 + dist*1e-3/vels_RW + tair, 1/periods_RW, c=cols[imode], ls=lss[imode], **label)

        ####################################################
        ax_mapb = ax_map.twinx()
        ax_map.set_title(f'Distance to source {dist/1e3:.1f} km', fontsize=fontsize)
        ax_map.legend(loc='lower center', bbox_to_anchor=(0.5, 1.06),ncol=nm+1, frameon=False)
        ax_map.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labelsize=fontsize)    
        ax_map.set_yscale("log")

        ax_mapb.set_ylabel('Period / [s]', fontsize=fontsize)
        ax_map.set_ylabel('Frequency / [Hz]', fontsize=fontsize)    
        ax_mapb.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labelsize=fontsize)     
        ax_mapb.set_yscale("log")

        ax_map.set_ylim(1/p_FTAN.max(), 1/p_FTAN.min() )
        ax_mapb.set_ylim(p_FTAN.max(), p_FTAN.min())
        ####################################################
        

        ####################################################
        ### Plot signal 
        ax = fig.add_subplot(grid[2,0], sharex=ax_map)
        ax.plot(time, signal, color='black', lw=1)
        ### Theoretical arrivals:
        th_P = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='p')
        th_S = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='s')
        ax_map.axvline(t0 + th_P + tair,color='k', ls='-', lw=1)
        ax_map.axvline(t0 + th_S + tair,color='k', ls='-', lw=1)
        ### Plot picks (do not add tair, it should have been included when picking)
        gauss_P = signal.max()*np.exp(-((time-arrival_time_P -t0)/std_P)**2)
        ax.plot(time, gauss_P, color='navy', label='P', lw=1)
        ax_map.axvline(t0 + arrival_time_P,color='navy', ls='--', lw=1)
        ###
        gauss_S = signal.max()*np.exp(-((time-arrival_time_S-t0)/std_S)**2)
        ax.plot(time, gauss_S, color='crimson', label='S', lw=1)
        ax_map.axvline(t0 + arrival_time_S,color='crimson', ls='--', lw=1)
        
        ####################################################
        ax.legend(frameon=False)
        ax.set_xlabel('Time / [s]', fontsize=fontsize)
        ax.set_ylabel('Vertical velocity / [m/s]', fontsize=fontsize)
        ax.set_xlim([time.min(), min(3*(arrival_time_S+t0),time.max())])
        ax.tick_params(axis='both', labelsize=fontsize)
        ####################################################
        ### Plot ticks in "group velocity" as well
        def inverse_ticks(x):
            return(1e-3*dist/(x+1e-10))
        axb = ax.twinx()
        ### To ensure ticks are ploted at max 8 km/s (very messy otherzise near t=0s)
        ax2 = axb.secondary_xaxis(location="top", functions = (inverse_ticks, inverse_ticks))
        ax2.set_xticks([8,7,6,5,4,3,2,1])
        ax2.set_xlabel('Group Velocity (km/s)', fontsize=fontsize)
        ####################################################
        fig.subplots_adjust(left=0.15, right=0.87, hspace=.7)
        fig.align_labels()
        # fig.savefig(".png".format(ievent), dpi=300)
        ####################################################


    #######################################################################################################
    def FTAN_and_arrival_times( self, waveform, dist,  ievent):

        print('Loading recorded data.')
        if self.key_traces == 1:   ### Obspy trace 
            ### Center waveforms on reference time
            otimes = waveform.times(reftime=self.t_ref)
            ### Cut a little before reference time 
            tmin = max(-2000, min(waveform.times(reftime=self.t_ref))) 
            wtime = otimes[otimes>tmin]
            dt = abs(wtime[1]-wtime[0])
            #waveform.detrend("polynomial", order=3)
            waveform_current_amp = waveform.data[otimes>tmin]
            waveform_current_amp *= windows.tukey(waveform_current_amp.size, alpha=0.02)
            
        else: ### Other type of trace (Ex Pyrocko synthetics) [NOT TESTED]
            wtime = waveform.get_xdata()[waveform.get_xdata()>0]
            waveform_current_amp = waveform.get_ydata().copy()[waveform.get_xdata()>0]
            dt = abs(wtime[1]-wtime[0])
            
        ### Calculate wave velocity from distance and arrival
        ### Don't forget to account for air propagation time 
        vels_orig = ((dist+self.depth)*1e-3)/(wtime-self.tprops_air[ievent])
        ### Calculate theoretical RW dispersion 
        periods_RW = np.logspace(np.log10(dt*2), np.log10(self.periods.max()), 1000)
        periods_RW_modes, vels_RW_modes = compute_vg_n_layers(periods_RW, self.true_velocity_model, max_mode = 3,)

        print('Computing FTAN.')
        ### Calculate FTAN with frequencies between the user-defined min frequency and Nyquist
        periods_FTAN = np.logspace(np.log10(dt*2), np.log10(self.periods.max()), self.periods.size)
        amplitude, phase = self.calculate_FTAN( waveform_current_amp, dt, periods_FTAN)
 
        ### Hand-picking the Rayleifh wave and P/S arrivals through an interactive Figure. 
        print('Start phase picker... ')
        freq_list = [1/self.periods.max(), 1/(2*dt)-0.01 ]
        arrival_time_P, arrival_time_S, std_P, std_S = self.pick_by_hand_PS_envelope(waveform, amplitude, 1/periods_FTAN, wtime, dist, ievent, freq_list=freq_list)
        ### Method with filter banks 
        freq_list = 10**np.linspace(np.log10(1/self.periods.max()), np.log10(1/(2*dt)-0.01), 10)
        arrival_time_P, arrival_time_S, std_P, std_S = self.pick_by_hand_PS_filterbank(waveform, dist, ievent, freq_list=freq_list)
        ### Method with FTAN 
        picked_t, picked_vg, picked_fg, std_RW, delta_vg = self.pick_by_hand_FTAN(amplitude, 1/periods_FTAN, vels_orig, wtime , ievent, 1e-3*(dist+self.depth),arrival_time_P, arrival_time_S)


        return amplitude, periods_FTAN, wtime, picked_t, picked_vg, picked_fg, delta_vg, periods_RW_modes, vels_RW_modes, arrival_time_P, arrival_time_S, std_RW, std_P, std_S


    #######################################################################################################
    def calculate_FTAN(self, x, dt, periods, alpha=2*np.pi**2, phase_corr=None):
        """
        Frequency-time analysis of a time series.
        Calculates the Fourier transform of the signal (xarray),
        calculates the analytic signal in frequency domain,
        applies Gaussian bandpass filters centered around given
        center periods, and calculates the filtered analytic
        signal back in time domain.
        Returns the amplitude/phase matrices A(f0,t) and phi(f0,t),
        that is, the amplitude/phase function of time t of the
        analytic signal filtered around period T0 = 1 / f0.

        See. e.g., Levshin & Ritzwoller, "Automated detection,
        extraction, and measurement of regional surface waves",
        Pure Appl. Geoph. (2001) and Bensen et al., "Processing
        seismic ambient noise data to obtain reliable broad-band
        surface wave dispersion measurements", Geophys. J. Int. (2007).

        @param dt: sample spacing
        @type dt: float
        @param x: data array
        @type x: L{numpy.ndarray}
        @param periods: center periods around of Gaussian bandpass filters
        @type periods: L{numpy.ndarray} or list
        @param alpha: smoothing parameter of Gaussian filter
        @type alpha: float
        @param phase_corr: phase correction, function of freq
        @type phase_corr: L{scipy.interpolate.interpolate.interp1d}
        @rtype: (L{numpy.ndarray}, L{numpy.ndarray})
        """

        ### Initializing amplitude/phase matrix: each column =
        ### amplitude function of time for a given Gaussian filter
        ### centered around a period
        amplitude = np.zeros(shape=(len(periods), len(x)))
        phase = np.zeros(shape=(len(periods), len(x)))

        xi = x.copy()
        x = x - np.mean(x)
        ### power of 2 nearest to N
        base2 = np.fix(np.log(x.size) / np.log(2) + 0.4999)
        nzeroes = (2 ** (base2 + 1) - x.size).astype(np.int64)
        x = np.concatenate((x, np.zeros(nzeroes)))

        ### Fourier transform
        Xa = fft(x)
        ### aray of frequencies
        freq = fftfreq(len(Xa), d=dt)

        ### analytic signal in frequency domain:
        #         | 2X(f)  for f > 0
        # Xa(f) = | X(f)   for f = 0
        #         | 0      for f < 0
        # with X = fft(x)
        Xa[freq < 0] = 0.0
        Xa[freq > 0] *= 2.0

        ### applying phase correction: replacing phase with given phase function of freq
        if phase_corr: ### [NOT TESTED]
            # doamin of definition of phase_corr(f)
            minfreq = phase_corr.x.min()
            maxfreq = phase_corr.x.max()
            mask = (freq >= minfreq) & (freq <= maxfreq)

            # replacing phase with user-provided phase correction:
            # updating Xa(f) as |Xa(f)|.exp(-i.phase_corr(f))
            phi = phase_corr(freq[mask])
            Xa[mask] = np.abs(Xa[mask]) * np.exp(-1j * phi)

            # tapering
            taper = cosTaper(npts=mask.sum(), p=0.05)
            Xa[mask] *= taper
            Xa[~mask] = 0.0

        # applying narrow bandpass Gaussian filters
        for iperiod, T0 in enumerate(periods):
            
            f0 = 1.0 / T0
            ### Gaussian filter (similar to S-transform)
            Xa_f0 = Xa * np.exp(-alpha * ((freq - f0) / f0) ** 2)

            ### IFFT: back to time domain
            xa_f0 = ifft(Xa_f0)

            ### Remove padding 
            xa_f0 = xa_f0[:xi.size]

            ### filling amplitude and phase in arrays
            amplitude[iperiod, :] = np.abs(xa_f0)
            phase[iperiod, :] = np.angle(xa_f0)
        
        return amplitude, phase


    ###############################################################################################################
    def pick_by_hand_FTAN(self, array, fr, vg, t, ievent, dd, t_P, t_S, finished=False, do_enhance = True, pick_in_time=True,
                            downsample=False, max_t_samples = 2000):

        def time_to_vel(t):
            return(dd/(t-self.tprops_air[ievent]))
        def vel_to_time(v):
            return(dd/v+self.tprops_air[ievent])

        plot_anyway = False 
        ### Careful: Not the same thing as ievent, here refers to the selected trace among 10 
        itrace = self.select[ievent]

        if os.path.isfile(self.data_dir + "Picks_event_{:d}_RW.npy".format(itrace)):
            freqcurve, tcurve, deltat, vgcurve, deltavg = np.load(self.data_dir + "Picks_event_{:d}_RW.npy".format(itrace))
            ### Ensure it is in descending order in freq
            tcurve = np.array([x for _, x in sorted(zip(freqcurve, tcurve), reverse=True)])
            deltat = np.array([x for _, x in sorted(zip(freqcurve, deltat), reverse=True)])
            vgcurve = np.array([x for _, x in sorted(zip(freqcurve, vgcurve), reverse=True)])
            deltavg = np.array([x for _, x in sorted(zip(freqcurve, deltavg), reverse=True)])
            freqcurve = np.array(sorted(freqcurve, reverse=True))

            finished = True
            plot_anyway = self.plot 
            if not plot_anyway:
                return(tcurve, vgcurve, freqcurve, deltat, deltavg)

        if not finished or plot_anyway:

            ### Remove parts that are too fast (12 km/s): 
            iv = np.where((vg>0) & (vg<12.))[0][0]
            ### Remove too negative times for time array (-100s): 
            ite = np.where(t>-100)[0][0]
            ### Remove later parts of signal 
            if t[-1]> 2*dd/2.5:
                itf = np.where(t>dd/1.)[0][0]
            else:
                itf = np.where(t>0)[0][-1]#None
            

            vg = vg[iv:itf]
            t = t[:itf]

            arrayt = array.copy()
            array = array[:,iv:itf]
            
            t = t[ite:itf]
            arrayt= arrayt[:,ite:itf]

            ### Center on rayleigh wave (not possible if group velocity not known):
            rw_1, rw_2 = 2., 5.
            rw_zone = np.where((vg>rw_1) & (vg<rw_2))[0]
            rw_zonet = np.where((t<vel_to_time(rw_1)) & (t>vel_to_time(rw_2)))[0]

            ### Easier vizualisation : scale by max at each frequency, trying to stay close to previous max. 
            array = array/np.max(array[:,rw_zone], keepdims=True, axis=1)
            arrayt = arrayt/np.max(arrayt[:,rw_zonet], keepdims=True, axis=1)

            ### High contrast 
            array = 10**abs(array)
            arrayt = 10**arrayt
            vmin, vmax =  10**0, 10**1 


            ifmax, itmax = np.where(arrayt == arrayt[:,rw_zonet].max())
            ifmax2, ivgmax = np.where(array == array[:,rw_zone].max())
            tampmax = t[itmax]
            vgampmax = vg[ivgmax] 
            fampmax, fampmax2 = fr[ifmax], fr[ifmax2]

            if downsample or arrayt.shape[1]>max_t_samples:
                print("Array downsampled")
                ### For faster plotting: time interpolation to have less than 5000 samples. 
                downsample_fact = max(5,int(arrayt.shape[1]//max_t_samples))
                fr, t, arrayt = block_mean(arrayt, fr, t, downsample_fact)
         
        ### PLOT INTERACTIVE PICKER 
        if not finished or plot_anyway:
            
            #fig, ax = plt.subplots(figsize=(9,5))
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1,1.5])
            
            ###############################################
            ### Ax for figure 
            axg = fig.add_subplot(gs[0,0])
            axt = fig.add_subplot(gs[1,0])
            if pick_in_time:
                ax = axt 
            else:
                ax = axg 
            ax.text(.99, .99, 'Pick here', ha='right', va='top', transform=ax.transAxes, color="Lime")

            cmap = cmc.lipari
            axg.pcolormesh(fr, vg, array.T, cmap =cmap, vmin=vmin, vmax=vmax)
            axt.pcolormesh(t, fr, arrayt, cmap =cmap, vmin=vmin, vmax=vmax)

            ### plot P and S pick before 
            if t_P !=0 :
                axt.axvline(t_P, c="w", ls="--", lw=1)
            if t_S !=0 : 
                axt.axvline(t_S, c="w", ls="--", lw=1)

            ### Plot maximum as a guide 
            axt.plot(tampmax, fampmax, color="pink", ls="-", lw=1)
            axg.plot(fampmax2, vgampmax, color="pink", ls="-", lw=1)

            axg.set_ylim(vg.min(),9)
            axg.set_xscale("log")
            axg.set_xlabel("Frequency / [Hz]")
            axg.set_ylabel("Group velocity (hypothesis) / [km/s]")
            ###
            axt.set_yscale("log")
            axt.set_xlim(t.min(), t.max())
            axt.set_ylim(fr.min(), fr.max())
            axt.set_ylabel("Frequency / [Hz]")
            axt.set_xlabel("Time after reference time / [s]")
            fig.align_labels()
            
            ### Plot results from previous time. 
            if finished:  
                tP = axg.errorbar(freqcurve, vgcurve, yerr = deltavg, c="grey", marker="+", ls="")
                tPt = axt.errorbar( tcurve, freqcurve, xerr = deltat, c="grey", marker="+", ls="")

            axg.set_title("Double-click once to select the most probable arrival,\nthen double-click once before and after to define uncertainty.")

            ##################################################
            ### Right subplot for list of points
            ax_list = fig.add_subplot(gs[0,1])
            ax_list.axis('off')  # Hide axes
            text_box = ax_list.text(0.05, 0.95, 'Selected Points in Group Velocity:', verticalalignment='top', 
                                    fontsize=10, wrap=True)
            ax_list_t = fig.add_subplot(gs[1,1])
            ax_list_t.axis('off')  # Hide axes
            text_box_t = ax_list_t.text(0.05, 0.95, 'Selected Points in Time:', verticalalignment='top', 
                                    fontsize=10, wrap=True)
            

            ###################################################
            ### Bottom area for the button
            ax_button = fig.add_axes([0.85, 0.02, 0.1, 0.05])
            button = Button(ax_button, 'Save & Exit')
            ax_undo = fig.add_axes([0.7, 0.02, 0.1, 0.05])
            button_undo = Button(ax_undo, 'Undo Last')

            
            ### Fill list of points, contains (frequency, time, delta_time, vg, delta_vg) 
            selected_points = []
            self.count_ev = 0
            points = [0,0,0]
            points_t = [0,0,0]
            temporaryPoints = []
            def onclick(event, temporaryPoints):
                #global temporaryPoints
                if event.inaxes != ax:
                    if event.inaxes == ax_button: 
                        selected_array = np.array(selected_points)
                        fval = selected_array[:,0]
                        tval = selected_array[:,1]
                        delta_t = selected_array[:,2]
                        vgval = selected_array[:,3]
                        delta_vg = selected_array[:,4]
                        file_name = self.data_dir + "Picks_event_{:d}_RW.npy".format(itrace)
                        np.save(file_name, [fval, tval, delta_t, vgval, delta_vg])
                        print("Points saved to '" + file_name + "'. Exiting.")
                        plt.close(fig)
                        return
                    elif event.inaxes == ax_undo and len(selected_points)>0: 
                        selected_points.pop()
                        update_text_box()
                        for i in range(2):
                            temporaryPoints[-1].remove()
                            temporaryPoints.pop()
                    return
                

                if event.dblclick:
                    if self.count_ev %3 == 0:
                        x_click = event.xdata
                        y_click = event.ydata  
                        if pick_in_time:
                            points_t[0] = y_click.copy()  # freq
                            points_t[1] = x_click.copy()  # time 
                            points[0] = points_t[0]
                            points[1] = time_to_vel(points_t[1])
                        else:
                            points[0] = x_click.copy()    # freq 
                            points[1] = y_click.copy()    # vg 
                            points_t[0] = points[0]
                            points_t[1] = vel_to_time(points[1])
                        # print(f"Clicked at x={x_click:.2f}, y={y_click:.2f}")
                        print("Click on upper bound for y.")
                    elif self.count_ev %3 == 1:
                        if pick_in_time:
                            points_t[2] = event.xdata.copy()
                        else:
                            points[2] = event.ydata.copy()
                        print("Click on lower bound for y.") 
                    elif self.count_ev %3 == 2: 
                        if pick_in_time:
                            x_minus = event.xdata.copy()
                            points_t[2] = abs((points_t[2]-x_minus)/2)
                            points[2] = abs(time_to_vel(points_t[1]- points_t[2]) - \
                                            time_to_vel(points_t[1]+ points_t[2]))/2
                        else:
                            y_minus = event.ydata.copy()
                            points[2] = abs((points[2]-y_minus)/2)
                            points_t[2] = abs(  vel_to_time(points[1]- points[2]) - \
                                                vel_to_time(points[1]+ points[2]))/2
                        ###    
                        selected_points.append(copy.deepcopy(points_t + points[1:]))
                        selected_points.sort(key=lambda point: point[0])
                        update_text_box()
                        tP = axg.errorbar(points[0], points[1], yerr = points[2], c="green", marker="+", ls="")
                        tPt = axt.errorbar( points_t[1], points_t[0], xerr = points_t[2], c="green", marker="+", ls="") 
                        temporaryPoints += [tP, tPt]

                    self.count_ev += 1 

            def update_text_box():
                text = 'Selected Points in Group Velocity:\n ' + 'f              vg           Delta_vg\n'
                text_t = 'Selected Points in Time:\n ' + 'f                 t               Delta_t\n'
                for idx, (p_f, p_t, p_dt, p_vg, p_dvg) in enumerate(selected_points, 1):
                    text += "{:.3f}      {:.2f}            {:.2f} \n".format(p_f, p_vg, p_dvg)
                    text_t += "{:>4.3f}      {:>6.2f}         {:>6.2f} \n".format(p_f, p_t, p_dt)
                text_box.set_text(text)
                text_box_t.set_text(text_t)
                fig.canvas.draw_idle()

            cid = fig.canvas.mpl_connect('button_press_event', lambda event : onclick(event, temporaryPoints) )
            # button.on_clicked(on_button_click)

            plt.show(block=True)

            ### Update values of freqcurve, etc
            ### Ensure it is in descending order in freq
            freqcurve, tcurve, deltat, vgcurve, deltavg = np.load(self.data_dir + "Picks_event_{:d}_RW.npy".format(itrace))
            tcurve = np.array([x for _, x in sorted(zip(freqcurve, tcurve), reverse=True)])
            deltat = np.array([x for _, x in sorted(zip(freqcurve, deltat), reverse=True)])
            vgcurve = np.array([x for _, x in sorted(zip(freqcurve, vgcurve), reverse=True)])
            deltavg = np.array([x for _, x in sorted(zip(freqcurve, deltavg), reverse=True)])
            freqcurve = np.array(sorted(freqcurve, reverse=True))

        return(tcurve, vgcurve, freqcurve, deltat, deltavg)


    ###############################################################################################################
    def pick_by_hand_PS_envelope(self, waveform, a_ftan, f_ftan, t_ftan, dist, ievent, twin = [5,10,20] , freq_list=[5e-3, 5], finished=False):
        plot_anyway = False
        ### Careful: not the same as ievent, 
        ### Here refers to the index of the trace among the "select" list 
        itrace = self.select[ievent]

        if os.path.isfile(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace)):
            P_pick, deltaP, S_pick, deltaS= np.load(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace))
            
            finished = True
            plot_anyway = self.plot
            if not plot_anyway:
                return(P_pick, S_pick, deltaP, deltaS)

          
        ### PLOT INTERACTIVE P-S PICKER 
        if not finished or plot_anyway:
            
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 3, width_ratios=[1,0.02,0.2], height_ratios = [1,1,2],
                                    left=0.1,right=0.92,top=0.93,bottom=0.08,wspace=0.15,hspace=0.15)
            axs = []
            axs.append(fig.add_subplot(gs[0, 0]) )
            axs.append(fig.add_subplot(gs[1, 0], sharex = axs[0]) )
            axs.append(fig.add_subplot(gs[2, 0], sharex = axs[0]) )
            axc = fig.add_subplot(gs[1, 1])
                    
            time = waveform.times(reftime=self.t_event)
            tmin, tmax = time.min(), time.max()
            dt = waveform.stats.delta
            ### Window after start time 
            it0 = np.where(time >0)[0][0]
            it1 = np.where(time >1000)[0][0]
            ###############################################

            ### PARAMETERS FOR BALLOONS VS SEISMIC 
            if self.tprops_air[ievent] == 0.0:
                fmin, fmax = 1, 1
                fbmin, fbmax = 2e-1,8e-1
            else:
                fmin, fmax = 0.05, 0.1
                fbmin, fbmax = 3e-2, 1e-1

            ### ORIGINAL SIGNAL 
            waveformb = waveform.copy()
            waveformc = waveform.copy()
            waveformb.detrend("linear")
            waveformb.taper(0.05, type="hann")
            waveformb.filter("bandpass", freqmin = min(freq_list), freqmax = max(freq_list), zerophase=True)
            waveformc.detrend("linear")
            waveformc.taper(0.05, type="hann")
            waveformc.filter("highpass", freq = fmin, zerophase=True)
            
            ### Plot Vel_Z or Pressure signal
            amin, amax = 0, 0
            amin , amax = waveformb.data.min() , waveformb.data.max() 
            axs[0].plot(time, waveformb.data, c="k", alpha = 1, lw=1, label=waveformb.stats.channel, ls = "-")
            axs[0].plot(time, waveformc.data, c="grey", alpha = 1, lw=1, ls = "-")
            ### Plot hilbert transform
            # axs[0].plot(time, np.abs(hilbert(waveformb[0].data)), c="b", lw=1, label="Hilbert envelope")
            axs[0].set_ylim(1.1*amin, 1.1*amax) 
            axs[0].set_title("Double-click to set P, then double-click before or after to set " + r"$\Delta$P" +". Same with S.\nEstimated distance: " + "d={:.0f} km".format(dist/1e3))

            ### DIFFERENT ENVELOPE FUNCTIONS 
            ### Some options filter out part of the signal 
            sig_filtered_low = waveform.copy()
            sig_filtered_low.detrend("linear")
            sig_filtered_low.taper(0.05, type="hann")
            sig_filtered_low.filter("lowpass", freq = fmax, zerophase=True)
            sig_filtered_high = waveform.copy()
            sig_filtered_high.detrend("linear")
            sig_filtered_high.taper(0.05, type="hann")
            sig_filtered_high.filter("highpass", freq = fmin, zerophase=True)
            sig_filtered_band = waveform.copy()
            sig_filtered_band.detrend("linear")
            sig_filtered_band.taper(0.05, type="hann")
            sig_filtered_band.filter("bandpass", freqmin = fbmin, freqmax = fbmax, zerophase=True)
            ### RMS Envelopes 
            rms_envelope = np.abs(waveformb.data)
            rms_envelope_flow = np.abs(sig_filtered_low.data)
            rms_envelope_fhigh = np.abs(sig_filtered_high.data)
            rms_envelope_fband = np.abs(sig_filtered_band.data)
            hilbert_envelope = np.abs(hilbert(waveformb.data))
            ### Hilbert 
            hilbert_envelope_flow = np.abs(hilbert(sig_filtered_low.data))
            hilbert_envelope_fhigh = np.abs(hilbert(sig_filtered_high.data))
            hilbert_envelope_fband = np.abs(hilbert(sig_filtered_band.data))

            ### SMOOTH ENVELOPE BY A TWIN SECOND WINDOW RUNNING MEAN 
            rms_envelope_l = [mean_before(rms_envelope, tw, dt) for tw in twin]
            rms_envelope_flow_l = [mean_before(rms_envelope_flow, tw, dt) for tw in twin]
            rms_envelope_fhigh_l = [mean_before(rms_envelope_fhigh, tw, dt) for tw in twin]
            rms_envelope_fband_l = [mean_before(rms_envelope_fband, tw, dt) for tw in twin]
            ###
            hilbert_envelope_l = [mean_before(hilbert_envelope, tw, dt) for tw in twin]
            hilbert_envelope_fhigh_l = [mean_before(hilbert_envelope_fhigh, tw, dt) for tw in twin]
            hilbert_envelope_fband_l = [mean_before(hilbert_envelope_fband, tw, dt) for tw in twin]
            hilbert_envelope_flow_l = [mean_before(hilbert_envelope_flow, tw, dt) for tw in twin]

            list_envelopes =rms_envelope_l + hilbert_envelope_l +\
                            rms_envelope_flow_l + hilbert_envelope_flow_l +\
                            rms_envelope_fband_l + hilbert_envelope_fband_l +\
                            rms_envelope_fhigh_l + hilbert_envelope_fhigh_l
            list_names = ["rms_{:d}".format(it) for it in range(len(twin))] + ["hil_{:d}".format(it) for it in range(len(twin))]+\
                          ["rms_Low_{:d}".format(it) for it in range(len(twin))] + ["hil_Low_{:d}".format(it) for it in range(len(twin))]+\
                          ["rms_band_{:d}".format(it) for it in range(len(twin))] + ["hil_band_{:d}".format(it) for it in range(len(twin))]+\
                          ["rms_hig_{:d}".format(it) for it in range(len(twin))] + ["hil_hig_{:d}".format(it) for it in range(len(twin))]   
            label_envelopes = ["RMS", "Hilbert", 
                            "RMS, f<{:.1f}".format(fmax), "Hilbert, f<{:.1f}".format(fmax),
                            "RMS, {:.1f}<f<{:.1f}".format(fbmin, fbmax), "Hilbert, {:.1f}<f<{:.1f}".format(fbmin, fbmax),
                            "RMS, f>{:.1f}".format(fmin),  "Hilbert, f>{:.1f}".format(fmin)]
            ne = len(list_envelopes)
            nm = 4 # Number of methods 
            cmap = plt.get_cmap("viridis")
            colse = []
            for i in range(4):
                colse += [cmap((i)/(nm-0.5))]# + [cmap((i)/(ne/2-.5)) for tw in twin]
            
            #######################################################################################3
            ### PLOT FTAN  
            p = np.abs(a_ftan)**2
            ta = np.where((t_ftan<=0) & (t_ftan>t_ftan.min()+300) )[0]
            p = p/np.median(p[:,ta], axis=1)[:,np.newaxis]
            ### Set the DB scale
            col = 10 * np.log10(p[:, :])
            ###
            wrf = np.where((f_ftan>1e-1) & (f_ftan<1e0))
            dBmin = np.mean(col[wrf]) - 3*np.std(col[wrf])
            dBmax = col[wrf].max()
            ###
            f_ftan, t_ftan, col = block_mean(col, f_ftan, t_ftan, 5)
            ###
            # ct = axs[1].pcolormesh(t_ftan, f_ftan, col,rasterized=True,
            #                     vmin= dBmin, vmax=dBmax, cmap=cmc.lipari,shading='auto')
            ### IMSHOW HAS MUCH BETTER PERFORMANCES ! CAREFUL WITH THE LOG AXIS THOUGH
            ct = axs[1].imshow(col,interpolation='nearest',aspect='auto',
                               extent=[t_ftan.min(), t_ftan.max(), np.log10(f_ftan.min()), np.log10(f_ftan.max())],
                                vmin= dBmin, vmax=dBmax, cmap=cmc.lipari,rasterized=True)
            axs[1].set_ylabel('Frequency / $Hz$')
            axs[1].set_ylim([np.log10(min(freq_list)), np.log10(max(freq_list))])
            axs[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(pow_10))
            #### 
            ### Colorbar 
            cb = plt.colorbar(mappable=ct, cax=axc)
            axc.set_ylabel(r'PSD / Median Noise PSD')
            axc.tick_params(axis='both', which='both')
            axc.yaxis.set_label_position('left')
            

            ################################################
            ### LOOP ON ENVELOPES 
            js = -1
            for j in range(4):
                nf = int(2*len(twin))
                for itw in range( nf ):
                    fe = list_envelopes[j*nf+itw]  
                    if itw ==0:
                        js +=1
                        axs[-1].plot(time, fe/fe.max() - js*.85, c=colse[j], label=label_envelopes[2*j], lw=1, alpha=0.5)
                    elif itw ==len(twin):
                        js +=1
                        axs[-1].plot(time, fe/fe.max() - js*.85, c=colse[j], label=label_envelopes[2*j+1], lw=1, alpha=0.5)
                    else:
                        axs[-1].plot(time, fe/fe.max() - js*.85, c=colse[j],  lw=1, alpha=0.5)
            axs[-1].axvline(0, color="k", ls='--')
            
        
            ################################################
            ### DECOR 
            axs[0].set_ylabel("Original\n")
            axs[0].set_facecolor('whitesmoke')
            axs[-1].set_xlabel("Time / [s]")
            axs[-1].text(0, -ne*0.8, "Ref. time", ha="center", va="top")
            for a in axs:
                a.set_xlim(time[it0]-100, time[it0]+1000)
                # a.get_xaxis().set_major_formatter(mdates.DateFormatter('%H:%M'))
            axs[-1].set_xlabel("Time after reference time / [$s$]")
            axs[-1].legend(loc=1, title="Envelope", edgecolor="none", framealpha=1)
            axs[0].legend(loc=1, edgecolor="none", framealpha=1)
            axs[-1].set_ylabel("Normalized envelopes")
            # axs[0].set_ylabel("Velocity / [$m/s$]")
            axs[0].set_ylabel("Signal")
            plt.setp(axs[-1].get_yticklabels(), visible=False)
            axs[-1].tick_params(left=False)

            #################################################
            ### Plot results from previous time. 
            for ax in axs:
                theory_P = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='p')
                theory_S = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='s')
                ax.axvline(theory_P + self.tprops_air[ievent], color="k", ls=":")
                ax.axvline(theory_S + self.tprops_air[ievent], color="k", ls=":")
                if finished:
                    ax.axvspan(P_pick-deltaP, P_pick+deltaP, color="navy", alpha=0.1)
                    ax.axvline(P_pick, color="navy", ls="--", alpha=0.3)
                    ax.axvspan(S_pick-deltaS, S_pick+deltaS, color="crimson", alpha=0.1)
                    ax.axvline(S_pick, color="crimson", ls="--", alpha=0.3)


            ##################################################
            ### Right subplot for list of points
            ax_list = fig.add_subplot(gs[0,2])
            ax_list.axis('off')  # Hide axes
            

            ###################################################
            ### Bottom area for the button
            ax_button = fig.add_axes([0.87, 0.1, 0.1, 0.05])
            button = Button(ax_button, 'Save & Exit')
            ax_undo = fig.add_axes([0.87, 0.17, 0.1, 0.05])
            button_undo = Button(ax_undo, 'Reset')
            ax_S = fig.add_axes([0.87, 0.25, 0.1, 0.05])
            button_S = Button(ax_S, 'Pick S')
            ax_P = fig.add_axes([0.75, 0.25, 0.1, 0.05])
            button_P = Button(ax_P, 'Pick P')
  
            ###################################################
            ### Fill list of picks [P, DP, S, DS] 
            if finished:
                points = [P_pick, deltaP, S_pick, deltaS]
            else:
                points = [0.,0.,0.,0.]
            text = '    P          '+r'$\Delta$P'+'          S        '+r'$\Delta$S' + '\n'
            text += "{: 4.1f}      {: 3.1f}      {: 4.1f}     {: 3.1f}\n".format(*points)
            text_box = ax_list.text(0.05, 0.95, text, verticalalignment='top', fontsize=10, wrap=True)    
            
            self.count_ev = 0
            tP1 = [ax.axvspan(-1e3,-1.1e3, color="navy", alpha=0.5) for ax in axs]
            tP2 = [ax.axvline(-1e3, color="navy", ls="-") for ax in axs]
            tP3 = [ax.axvspan(-1e3,-1.1e3, color="crimson", alpha=0.5) for ax in axs]
            tP4 = [ax.axvline(-1e3, color="crimson", ls="-") for ax in axs]


            def onclick(event, points, tP1, tP2, tP3, tP4 ):   

                if event.inaxes not in axs :
                    if event.inaxes == ax_button: 
                        array = np.array(points)
                        file_name = self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace)
                        np.save(file_name, array)
                        print("Points saved to '" + file_name + "'. Exiting.")
                        plt.close(fig)
                        return
                    elif event.inaxes == ax_undo: 
                        if finished:
                            points = [P_pick, deltaP, S_pick, deltaS]
                        else:
                            points = [0.,0.,0.,0.]
                        self.count_ev = 0
                        update_text_box(points)
                        for ia, ax in enumerate(axs):
                            _arr_tP1 = tP1[ia].get_xy()
                            _arr_tP1[:, 0] = [-1e3,-1e3, -1.1e3, -1.1e3,-1e3]
                            tP1[ia].set_xy(_arr_tP1)
                            tP2[ia].set_xdata([-1e3,-1e3])
                            _arr_tP3 = tP3[ia].get_xy()
                            _arr_tP3[:, 0] = [-1e3,-1e3, -1.1e3, -1.1e3,-1e3]
                            tP3[ia].set_xy(_arr_tP3)
                            tP4[ia].set_xdata([-1e3,-1e3])
                    elif event.inaxes == ax_P: 
                        self.count_ev = 0 
                    elif event.inaxes == ax_S: 
                        self.count_ev = 2
                        print("Double-click on S center")
                    return

                if event.dblclick: 
                    if self.count_ev%4 == 0:
                        print("Double-click on P center")
                        x_click = event.xdata
                        points[0] = x_click.copy() 
                        # print(f"Clicked at x={x_click:.2f}, y={y_click:.2f}")
                        print("Double-click on Pminus or Pplus")
                    elif self.count_ev%4 == 1:
                        x_diff = event.xdata
                        points[1] = abs(points[0]-x_diff)
                        
                        update_text_box(points)
                        for ia, ax in enumerate(axs):
                            _arr_tP1 = tP1[ia].get_xy()
                            _arr_tP1[:, 0] = [points[0]-points[1],points[0]-points[1], points[0]+points[1], points[0]+points[1],points[0]-points[1]]
                            tP1[ia].set_xy(_arr_tP1)
                            tP2[ia].set_xdata([points[0], points[0]])
                        print("Double-click on S center")

                    elif self.count_ev%4 == 2:
                        x_click = event.xdata
                        points[2] = x_click.copy() 
                        print("Double-click on Sminus or Splus")
                    elif self.count_ev%4 == 3:
                        x_diff = event.xdata
                        points[3] = abs(points[2]-x_diff)

                        update_text_box(points)
                        for ia, ax in enumerate(axs):
                            _arr_tP3 = tP3[ia].get_xy()
                            _arr_tP3[:, 0] = [points[2]-points[3],points[2]-points[3], points[2]+points[3], points[2]+points[3],points[2]-points[3]]
                            tP3[ia].set_xy(_arr_tP3)
                            tP4[ia].set_xdata([points[2], points[2]])
                       
                    # print(self.count_ev, points)
                    self.count_ev += 1 

            def update_text_box(points):
                text = '    P          '+r'$\Delta$P'+'          S        '+r'$\Delta$S' + '\n'
                text += "{: 4.1f}      {: 3.1f}      {: 4.1f}     {: 3.1f}\n".format(*points)
                text_box.set_text(text)
                fig.canvas.draw_idle()    

            cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, points, tP1, tP2, tP3, tP4 ) )        

            fig.align_labels()
            plt.show(block=True)

            ### Update values of freqcurve, etc
            ### Ensure it is in descending order in freq
            P_pick, deltaP, S_pick, deltaS = np.load(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace))

        fig.clf()
        
        return(P_pick, S_pick, deltaP, deltaS)


    ###############################################################################################################
    def pick_by_hand_PS_filterbank(self, waveform, dist, ievent, freq_list=[5e-3, 1e-2, 5e-2, 1e-1, 2e-1,5e-1, 1, 2, 5], finished=False):

        plot_anyway = False 
        NF = len(freq_list)-1
        ### Frequencies in decreasing order
        freq_list = sorted(freq_list)[::-1]
        ### Careful: not the same as ievent, refers to the index of the trace among 10 alaska
        itrace = self.select[ievent]

        if os.path.isfile(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace)):
            P_pick, deltaP, S_pick, deltaS= np.load(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace))
            
            finished = True
            plot_anyway = self.plot 
            if not plot_anyway:
                return(P_pick, S_pick, deltaP, deltaS)

        ### PLOT INTERACTIVE P-S PICKER 
        if not finished or plot_anyway:
            
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(NF, 2, width_ratios=[5, 1])
            
            time = waveform.times(reftime=self.t_event)
            ### Window after start time 
            it0 = np.where(time >0)[0][0]
            it1 = np.where(time >1000)[0][0]
            ###############################################
            ### Plot top (full signal)
            axs = []
            ax0 = fig.add_subplot(gs[0,0])
            axs.append(ax0)

            waveformb = waveform.copy()


            waveformb.detrend("linear")
            waveformb.taper(0.05, type="hann")
            waveformb.filter("bandpass", freqmin = min(freq_list), freqmax = max(freq_list), zerophase=True)
            ax0.plot(time, waveformb.data,  c='k', lw=1)
            ax0.set_xlim(time.min(), time.max())     
            ax0.set_title("Double-click to set P, then double-click before or after\nto set DeltaP. Same with S.\nestimated dist: " + "d={:.0f} km".format(dist/1e3))

            ################################################
            ### LOOP ON BANK 
            for i in range(NF-1):
                axi = fig.add_subplot(gs[i+1,0], sharex = ax0)
                axs.append(axi)
                waveformb = waveform.copy()
                waveformb.detrend("linear")
                waveformb.taper(0.05, type="hann")
                waveformb.filter("bandpass", freqmin=freq_list[i+1], freqmax=freq_list[i], zerophase=True)

                ### Plot pressure 
                axi.axvline(0, color="k", ls="-", alpha=0.3)
                axi.plot(time, waveformb.data/np.abs(waveformb.data[it0:it1]).max(), c='k', lw=1)
                axi.set_xlim(time[it0]-100, time[it0]+1000)
                axi.set_ylabel("{:.2g}-{:.2g} Hz".format(freq_list[i+1],freq_list[i]), rotation=0, ha='right')

            ################################################
            ### DECOR 
            axs[0].set_ylabel("Original\n")
            axs[0].set_facecolor('whitesmoke')
            axs[-1].set_xlabel("Time / [s]")
            axs[-1].text(0, -3, "Ref. time", ha="center", va="top")
            for ax in axs:
                ### Hide most spines and axes
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if ax != axs[-1]:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(bottom=False)
                if ax !=axs[0]:
                    ax.set_yticks([])
                    ax.set_ylim(-1,1)

            #################################################
            ### Plot results from previous time. 
            for ax in axs:
                theory_P = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='p')
                theory_S = compute_travel_time_SP(dist/1e3, self.depth/1e3, self.true_velocity_model, phase='s')
                ax.axvline(theory_P + self.tprops_air[ievent], color="k", ls=":")
                ax.axvline(theory_S + self.tprops_air[ievent], color="k", ls=":")
                if finished:
                    ax.axvspan(P_pick-deltaP, P_pick+deltaP, color="navy", alpha=0.1)
                    ax.axvline(P_pick, color="navy", ls="--", alpha=0.3)
                    ax.axvspan(S_pick-deltaS, S_pick+deltaS, color="crimson", alpha=0.1)
                    ax.axvline(S_pick, color="crimson", ls="--", alpha=0.3)

            ##################################################
            ### Right subplot for list of points
            ax_list = fig.add_subplot(gs[0,1])
            ax_list.axis('off')  # Hide axes
            

            ###################################################
            ### Bottom area for the button
            ax_button = fig.add_axes([0.85, 0.1, 0.1, 0.05])
            button = Button(ax_button, 'Save & Exit')
            ax_undo = fig.add_axes([0.85, 0.17, 0.1, 0.05])
            button_undo = Button(ax_undo, 'Reset')
            ax_S = fig.add_axes([0.85, 0.25, 0.1, 0.05])
            button_S = Button(ax_S, 'Pick S')
            ax_P = fig.add_axes([0.73, 0.25, 0.1, 0.05])
            button_P = Button(ax_P, 'Pick P')
            
            ### Fill list of picks [P, DP, S, DS] 
            if finished:
                points = [P_pick, deltaP, S_pick, deltaS]
            else:
                points = [0.,0.,0.,0.]
            text = '    P          '+r'$\Delta$P'+'          S        '+r'$\Delta$S' + '\n'
            text += "{: 4.1f}      {: 3.1f}      {: 4.1f}     {: 3.1f}\n".format(*points)
            text_box = ax_list.text(0.05, 0.95, text, verticalalignment='top', fontsize=10, wrap=True)    
            #####
            self.count_ev = 0
            tP1 = [ax.axvspan(-1e3,-1.1e3, color="navy", alpha=0.5) for ax in axs]
            tP2 = [ax.axvline(-1e3, color="navy", ls="-") for ax in axs]
            tP3 = [ax.axvspan(-1e3,-1.1e3, color="crimson", alpha=0.5) for ax in axs]
            tP4 = [ax.axvline(-1e3, color="crimson", ls="-") for ax in axs]
            temporaryPoints = []

            def onclick(event, points, tP1, tP2, tP3, tP4):   

                if event.inaxes not in axs :
                    if event.inaxes == ax_button: 
                        array = np.array(points)
                        file_name = self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace)
                        np.save(file_name, array)
                        print("Points saved to '" + file_name + "'. Exiting.")
                        plt.close(fig)
                        return
                    elif event.inaxes == ax_undo: 
                        if finished:
                            points = [P_pick, deltaP, S_pick, deltaS]
                        else:
                            points = [0.,0.,0.,0.]
                        self.count_ev = 0
                        update_text_box(points)
                        ### Reset axes containing markers 
                        for ia, ax in enumerate(axs):
                            _arr_tP1 = tP1[ia].get_xy()
                            _arr_tP1[:, 0] = [-1e3,-1e3, -1.1e3, -1.1e3,-1e3]
                            tP1[ia].set_xy(_arr_tP1)
                            tP2[ia].set_xdata([-1e3,-1e3])
                            _arr_tP3 = tP3[ia].get_xy()
                            _arr_tP3[:, 0] = [-1e3,-1e3, -1.1e3, -1.1e3,-1e3]
                            tP3[ia].set_xy(_arr_tP3)
                            tP4[ia].set_xdata([-1e3,-1e3])
                        # fig.canvas.draw_idle()
                    elif event.inaxes == ax_P: 
                        self.count_ev = 0 
                    elif event.inaxes == ax_S: 
                        self.count_ev = 2
                        print("Double-click on S center")
                    return

                if event.dblclick: 
                    if self.count_ev%4 == 0:
                        print("Double-click on P center")
                        x_click = event.xdata
                        points[0] = x_click.copy() 
                        print("Double-click on Pminus or Pplus")
                    elif self.count_ev%4 == 1:
                        x_diff = event.xdata
                        points[1] = abs(points[0]-x_diff)
                        
                        update_text_box(points)
                        for ia, ax in enumerate(axs):
                            _arr_tP1 = tP1[ia].get_xy()
                            _arr_tP1[:, 0] = [points[0]-points[1],points[0]-points[1], points[0]+points[1], points[0]+points[1],points[0]-points[1]]
                            tP1[ia].set_xy(_arr_tP1)
                            tP2[ia].set_xdata([points[0], points[0]])
                            
                        print("Double-click on S center")

                    elif self.count_ev%4 == 2:
                        x_click = event.xdata
                        points[2] = x_click.copy() 
                        print("Double-click on Sminus or Splus")
                    elif self.count_ev%4 == 3:
                        x_diff = event.xdata
                        points[3] = abs(points[2]-x_diff)

                        update_text_box(points)
                        for ia, ax in enumerate(axs):
                            _arr_tP3 = tP3[ia].get_xy()
                            _arr_tP3[:, 0] = [points[2]-points[3],points[2]-points[3], points[2]+points[3], points[2]+points[3],points[2]-points[3]]
                            tP3[ia].set_xy(_arr_tP3)
                            tP4[ia].set_xdata([points[2], points[2]])
                    
                    self.count_ev += 1 

            def update_text_box(points):
                text = '    P          '+r'$\Delta$P'+'          S        '+r'$\Delta$S' + '\n'
                text += "{: 4.1f}      {: 3.1f}      {: 4.1f}     {: 3.1f}\n".format(*points)
                text_box.set_text(text)
                fig.canvas.draw_idle() 

            cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, points, tP1, tP2, tP3, tP4))#temporaryPoints) )        

            plt.show(block=True)

            ### Update values of freqcurve, etc
            ### Ensure it is in descending order in freq
            P_pick, deltaP, S_pick, deltaS = np.load(self.data_dir + "Picks_event_{:d}_PS.npy".format(itrace))

        fig.clf()
        return(P_pick, S_pick, deltaP, deltaS)



    
########################################################################################################
### CLASS FOR RUNNING THE INVERSION
########################################################################################################
class MCMC_Model():

    #####################################################################################
    def __init__(self, DATA, 
                    run_name, save_dir, data_dir, param_file, method="emcee", n_iter=1e3, 
                    n_cpus=1, progress=True, do_mpi=False,
                    n_temp=1, n_swaps=100, n_save=10000,
                    use_backend_file=None, reset_backend=False, do_analysis = False):


        # Local dictionary of all local variables (arguments)
        locals_dict = locals()
        # Iterate through each item in the dictionary
        for name, value in locals_dict.items():
            if name != "self":
                # Set each as an attribute of the instance
                setattr(self, name, value)  

        ### Get information for simulation and priors:
        self.get_simulation_parameters()

        self.n_dim = len(self.prior_inv)
        ## Create initial conditions
        if not hasattr(self, 'pos') or self.pos is None:
            self.pos = generate_initial_model(self.n_dim,self.n_temp, self.n_walk, self.start_lims, self.wr_order, self.prior_lims, self.is_inverted)

        ### Arguments passed to log_likelihood calculation 
        self.args_MCMC = (  self.DATA.data_vector, self.DATA.std_vector, 
                            self.DATA.sta_lats, self.DATA.sta_lons,  self.DATA.tprops_air,
                            self.DATA.periods_RWs)

        if not do_analysis:
            self._load_sampler()

            self.run_finished = False
            self.solutions_postprocessed = False

        else: 
            self._load_results()
            self.run_finished = True
            


    #####################################################################################
    def get_simulation_parameters(self):

        #########################################################
        ### Parameters for inversion methods 
        if self.method == "emcee" or self.method == "emcee2":
            ### Number of Walkers for emcee (must be twice the number of dimensions)
            self.n_walk=25
            ### How often to save : 
            self.n_save = 10000

        elif self.method == "ptmcmc" :
            ### Number of temperatures
            self.n_temp=10#0
            #### Frequency of swaps :
            self.n_swaps = 100
            #### Number of walkers (= distinct parallel runs for ptmcmc)
            self.n_walk= self.n_cpus
            ### How often to save : 
            self.n_save = 100#00

            ### Parameter for the gaussian model generation:
            self.proportion = 50 

            ### Parameter for temperature
            self.tempering = False
            self.use_adapt = False
            self.do_swap   = False
            if self.n_temp !=1:
                self.tempering = True
                self.use_adapt = False 
                self.do_swap   = True
            ### Temperature ladder method from ptemcee
            tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                                2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                                2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                                1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                                1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                                1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                                1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                                1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                                1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                                1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                                1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                                1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                                1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                                1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                                1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                                1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                                1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                                1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                                1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                                1.26579, 1.26424, 1.26271, 1.26121,
                                1.25973])
            Tmax            = 1e4#tstep ** (NTP - 1)
            self.adapt_nu   =  100          # default = 100 (ptemcee)
            self.adapt_tlag = 10000       # default = 10000 (ptemcee)
            self.numtemp = range(self.n_temp)
            


        #########################################################
        ### Parameters to invert and prior bounds are defined in: 
        pdat = np.genfromtxt(self.param_file, delimiter='', skip_header=2,
                            dtype = ['U18', 'bool', 'U30', 'float64', 'float64', 'float64', 'float64'], 
                            names=['var', 'vinv', 'vname', 'prior_min', 'prior_max', 'start_min', 'start_max'])

        ### Check that the right number of layers and seismic parameters have been defined 
        n_vs = (pdat['var']=='vs').sum()# & (pdat['vinv']==True)).sum()
        n_ps = (pdat['var']=='poisson').sum()# & (pdat['vinv']==True)).sum()
        n_layers = (pdat['var']=='h_layer').sum()# & (pdat['vinv']==True)).sum()
        if n_vs>=2:
            if n_ps != n_vs :
                raise Warning("Mismatch of vs and Poisson ratio parameters")
            if n_layers != n_vs-1 :
                raise Warning("Mismatch of vs and nlayers parameters")
        else:
            raise ValueError("Single-layer inversion not implemented")
                
        ### Focus on variables that are inverted. We will construct the parameter vector
        pdat_inv = pdat[pdat['vinv']] #np.delete(pdat, np.where(pdat['vinv']==False),axis=0)
        NM = pdat_inv.size
        if self.method=="emcee" or self.method=="emcee2":
            ### Ensure right number of walkers for EMCEE
            if 2*NM > self.n_walk:
                self.n_walk = 2*NM+1
        if self.method=="ptmcmc":
            ### Create temperature scale for ptmcmc
            tstep = tstep[NM-1]
            self.Tladder = 10**(np.linspace(0, np.log10(Tmax), self.n_temp))
            # print("Temperature ladder : ", self.Tladder)

        ### Identify position of all parameters in list 
        wrall_ts = np.where(pdat['var'] == 'time_source')[0][0]
        wrall_Ls = [np.where(pdat['var'] == 'lat_source')[0][0], np.where(pdat['var'] == 'lon_source')[0][0]]
        wrall_hs = np.where(pdat['var'] == 'depth_source')[0][0]
        wrall_vs = np.where(pdat['var'] == 'vs')[0]
        wrall_ps = np.where(pdat['var'] == 'poisson')[0]
        wrall_hl = np.where(pdat['var'] == 'h_layer')[0]
        ### Identify position of each inverted parameter in list 
        if pdat[wrall_ts]["vinv"]:
            wr_ts = np.where(pdat_inv['var'] == 'time_source')[0][0]
        else:
            wr_ts = []
        if pdat[wrall_Ls[0]]["vinv"]:
            wr_lat = np.where(pdat_inv['var'] == 'lat_source')[0][0]
        else:
            wr_lat = []
        if pdat[wrall_Ls[1]]["vinv"]:
            wr_lon = np.where(pdat_inv['var'] == 'lon_source')[0][0]
        else:
            wr_lon = []  
        wr_Ls = [wr_lat, wr_lon]
        if pdat[wrall_hs]["vinv"]:
            wr_hs = np.where(pdat_inv['var'] == 'depth_source')[0][0]
        else:
            wr_hs = []
        wr_vs = np.where(pdat_inv['var'] == 'vs')[0]
        wr_ps = np.where(pdat_inv['var'] == 'poisson')[0]
        wr_hl = np.where(pdat_inv['var'] == 'h_layer')[0]

        ### Prepare list of names and labels:
        self.names = pdat_inv['vname']
        self.names[wr_ts] = r"$\Delta t_{source}$ [s]"
        self.names[wr_Ls] = [r"$Lat_{source}$ [$^{\circ}$]", r"$Lon_{source}$ [$^{\circ}$]"]
        self.names[wr_hs] = r"$h_{source}$ [km]"
        self.names[wr_vs] = [r"$v_s$"+ "{:d}".format(n) + r" [km$\cdot$s$^{-1}$]" for n in range(n_vs) if pdat[pdat['var']=='vs']['vinv'][n]]
        self.names[wr_ps] = [r"$\nu$"+ "{:d}".format(n) for n in range(n_ps) if pdat[pdat['var']=='poisson']['vinv'][n]]
        self.names[wr_hl] = [r"$H$" + "{:d}".format(n) +  " [km]" for n in range(n_layers) if pdat[pdat['var']=='h_layer']['vinv'][n]]

        ### Prepare list of prior bounds
        prior_min = pdat['prior_min']
        prior_max = pdat['prior_max']
        self.prior_lims = list(zip(prior_min,prior_max))
        self.prior_inv = [pi for pi,iv in zip(self.prior_lims, pdat['vinv']) if iv]
        self.n_blobs = len(self.prior_lims)- len(self.prior_inv)

        ### Prepare list of starting bounds 
        start_min = pdat_inv['start_min']
        start_max = pdat_inv['start_max']
        ### Ensure starting bounds are not outside of prior bounds
        wlow = np.where((start_min-pdat_inv['prior_min'])<0)
        start_min[wlow] = pdat_inv['prior_min'][wlow]
        whigh = np.where((pdat_inv['prior_max']-start_max)<0)
        start_max[whigh] = pdat_inv['prior_max'][whigh]  
        if wlow[0].size or whigh[0].size:
            print("Some starting bounds are outside of prior bouds. Correcting...")
        ### 
        self.start_lims = list(zip(start_min,start_max))
        ### To start with the widest possible boundaries
        if self.method == "ptmcmc":
            ### Define the stepsize for the Gaussian generation of new models. 
             self.stepsz = np.array([(pma -pmi)/self.proportion for pma, pmi in zip(start_max, start_min) ] )

        ### Final lists: Identify where each variable is inside the model parameter vector
        self.wr_order = [wr_ts, wr_Ls, wr_hs, wr_vs, wr_ps, wr_hl]
        ### Also identify where all variables are once sorted 
        self.wrall_order = [wrall_ts, wrall_Ls, wrall_hs, wrall_vs, wrall_ps, wrall_hl]
        ### Identify in all source and subsurface parameters which ones are inverted (usufull when plotting)
        self.any_inverted= pdat["vinv"]
        ### Same thing but in the format of wr_order, separating each different kind of variables  
        self.is_inverted = [pdat[pdat['var']=='time_source']['vinv']] + \
                                [[pdat[pdat['var']=='lat_source']['vinv'], pdat[pdat['var']=='lon_source']['vinv']]] + \
                                [pdat[pdat['var']==nvar]['vinv'] for nvar in ['depth_source', 'vs', 'poisson', 'h_layer']]
        ### We also store the priors sorted by variables   
        self.prior_min_sorted = [prior_min[pdat['var']=='time_source']] + \
                                [[prior_min[pdat['var']=='lat_source'], prior_min[pdat['var']=='lon_source']]] + \
                                [ prior_min[pdat['var']==nvar] for nvar in ['depth_source', 'vs', 'poisson', 'h_layer']]
        self.prior_max_sorted = [prior_max[pdat['var']=='time_source']] + \
                                [[prior_max[pdat['var']=='lat_source'], prior_max[pdat['var']=='lon_source']]] + \
                                [ prior_max[pdat['var']==nvar] for nvar in ['depth_source', 'vs', 'poisson', 'h_layer']]
        return 


    #####################################################################################
    def _init_backend(self, nwalkers, ndim, ndim_blobs=None, overwrite=False, irun=1):
        
        ### Name the backend file  
        if self.method == 'emcee2' : 
            self.filename = self.save_dir + "chains_emcee2.h5"
        elif self.method == 'ptmcmc' : 
            self.filename = self.save_dir + "chains_ptmcmc_{:d}.h5".format(irun)

        ### If resetting the backend, create an empty file 
        ### And fill it with the variables we want to save 
        if overwrite:
            print("Overwriting previous backend...")
            with h5py.File(self.filename, "w") as f:
                pass  # Create an empty file
            
            with h5py.File(self.filename, "a") as f:
                f.create_dataset("chain", shape=(0, nwalkers, ndim), maxshape=(None, nwalkers, ndim))
                f.create_dataset("log_prob", shape=(0, nwalkers), maxshape=(None, nwalkers))
                f.create_dataset("accepted", shape=(0, nwalkers), maxshape=(None, nwalkers))
            

    #####################################################################################
    def _append_backend(self, chain_buffer, log_prob_buffer, accepted_buffer):#, blobs_buffer):
        
        ### Append buffered data to the HDF5 datasets.
        with h5py.File(self.filename, "a") as f:
            # 1. Extend datasets 2. append 
            f["chain"].resize(f["chain"].shape[0] + chain_buffer.shape[0], axis=0)
            f["chain"][-chain_buffer.shape[0]:] = chain_buffer

            f["log_prob"].resize(f["log_prob"].shape[0] + log_prob_buffer.shape[0], axis=0)
            f["log_prob"][-log_prob_buffer.shape[0]:] = log_prob_buffer

            f["accepted"].resize(f["accepted"].shape[0] + accepted_buffer.shape[0], axis=0)
            f["accepted"][-accepted_buffer.shape[0]:] = accepted_buffer


    #####################################################################################
    ### CODE FOR CUSTOM MADE HDF5
    def _get_value(self, arr, flat=False, thin=1, discard=0):
        ### Thining from emcee
        v = arr[discard + thin - 1 :  : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v


    #####################################################################################
    def _load_sampler(self):

        #############################################################
        ### Create multiprocessing pool 
        if self.n_cpus>1:
            if self.do_mpi:
                self.pool = MPIPool()
                print("Running MPIPool...")
            else:
                self.pool = Pool(processes=self.n_cpus)        
                ncpu = cpu_count()
                print("{0} CPUs available".format(ncpu))
                print("{0} CPUs running".format(self.pool._processes))
        else:
            print("No pool, running on 1 CPU.")

        #############################################################         
        if self.method == 'emcee' :
            print("doing emcee...") 
 
            self.backend = None
            backend_dict = dict()
            if self.use_backend_file is not None:
                #self.backend = emcee.backends.HDFBackend(self.use_backend_file)
                self.filename = self.save_dir + self.use_backend_file
            else:
                self.filename = self.save_dir + "chains_emcee.h5"
            if self.reset_backend:
                ### For some reason "reset" isn't enough to delete the file. Better to simply delete it. 
                ###self.backend.reset(self.n_walk, self.n_dim)
                try:
                    os.remove(self.filename)
                    print("Deleting previous backend...")
                except OSError:
                    pass
                try: 
                    os.remove(self.save_dir + 'chains_emcee_c.h5')
                except OSError:
                    pass 
                try:
                    os.remove(self.save_dir + 'acceptance_rate.npy')
                except OSError:
                    pass 
                self.backend = emcee.backends.HDFBackend(self.filename)
                backend_dict['backend'] = self.backend
            else:
                self.backend = emcee.backends.HDFBackend(self.filename)
            
            ### NOTE: Better to avoid passing heavy args to emcee, such as in args_MCMC
            if self.n_cpus>1:
                self.sampler = emcee.EnsembleSampler(self.n_walk, self.n_dim, 
                                                     log_probability,args=(self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC),
                                                     backend=self.backend, pool=self.pool,
                                                     ### Note: classic move is Stretchmove only
                                                     ### Best adapted for highly correlated distributions
                                                     ### All other moves have a lower acceptance rate 
                                                     ### No significant improvement of convergence after 1e4 iterations
                                                    #  moves=[ (emcee.moves.DEMove(), 0.4),
                                                    #         (emcee.moves.StretchMove(), 0.4),
                                                    #         (emcee.moves.DESnookerMove(), 0.2) ],
                                                    #  moves=[(emcee.moves.StretchMove(), 1.0) ],
                                                    #  moves=[(emcee.moves.StretchMove(), 0.5), 
                                                    #         (emcee.moves.DEMove(), 0.3),
                                                    #         (emcee.moves.WalkMove(), 0.2)],
                                                     blobs_dtype=[("fixed_var_{:d}".format(i), float) for i in range(self.n_blobs)]
                                                    )#
            else:
                ### Version without pool, useful for debug with cprofile 
                self.sampler = emcee.EnsembleSampler(self.n_walk, self.n_dim, 
                                                     log_probability,args=(self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC),
                                                     backend=self.backend,
                                                    #  moves=[ (emcee.moves.DEMove(), 0.8),
                                                    #         (emcee.moves.StretchMove(), 0.4),
                                                    #         (emcee.moves.DESnookerMove(), 0.2) ],
                                                    #  moves=[ (emcee.moves.StretchMove(), 1.0) ],
                                                    #   moves=[(emcee.moves.StretchMove(), 0.5), 
                                                    #         (emcee.moves.DEMove(), 0.3),
                                                    #         (emcee.moves.WalkMove(), 0.2)],
                                                     blobs_dtype=[("fixed_var_{:d}".format(i), float) for i in range(self.n_blobs)]
                                                    )#

            try:
                state = self.sampler._previous_state
                self.allow_warm_start = True
            except:
                self.allow_warm_start = False
                
        #############################################################         
        elif self.method == 'emcee2' :
            print("doing emcee no backend...") 
 
            ### Create the backend file with emcee conventions 
            self._init_backend(self.n_walk, self.n_dim, overwrite = self.reset_backend)
            
            if self.n_cpus>1:
                self.sampler = emcee.EnsembleSampler(self.n_walk, self.n_dim, 
                                                     log_probability,args=(self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC),
                                                     pool=self.pool,
                                                     blobs_dtype=[("fixed_var_{:d}".format(i), float) for i in range(self.n_blobs)]
                                                    )#
            else:
                ### No pool for debug 
                self.sampler = emcee.EnsembleSampler(self.n_walk, self.n_dim, 
                                                     log_probability,args=(self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC),
                                                     blobs_dtype=[("fixed_var_{:d}".format(i), float) for i in range(self.n_blobs)]
                                                    )#

            try:
                ### Try opening chain file 
                with h5py.File(self.filename, "r") as f:
                    pass 
                self.allow_warm_start = True
            except:
                self.allow_warm_start = False
                
        
        #############################################################       
        # elif self.method == 'ptmcmc':
        #     print("doing Parallel Tempering...")
            
        #     self._init_backend(self.n_temp, self.n_dim, overwrite=self.reset_backend, n_cpus=self.n_cpus) 

        #     try:
        #         ### Try opening chain file 
        #         with h5py.File(self.filename, "r") as f:
        #             pass 
        #         self.allow_warm_start = True
        #     except:
        #         self.allow_warm_start = False


    #####################################################################################
    def _load_results(self):

        ###########################
        if self.method == 'emcee' :
            if self.use_backend_file is not None:
                #self.backend = emcee.backends.HDFBackend(self.use_backend_file)
                self.filename = self.save_dir + self.use_backend_file
            else:
                self.filename = self.save_dir + "chains_emcee_c.h5"
            self.backend = emcee.backends.HDFBackend(self.filename)

            ### Initialises sampler and blobs 
            self.sampler = emcee.EnsembleSampler(self.n_walk, self.n_dim, 
                                                 log_probability,args=(self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC),
                                                 backend=self.backend,
                                                 blobs_dtype=[("fixed_var_{:d}".format(i), float) for i in range(len(self.prior_lims)- len(self.prior_inv))]
                                                 )
            
        ###########################
        if self.method == 'emcee2' :
            self.filename = self.save_dir + "chains_emcee2_c.h5"
            

        #############################      
        elif self.method == 'ptmcmc':
            self.filenames = [self.save_dir + f for f in os.listdir(self.save_dir) if f.startswith("chains_ptmcmc_") and f.endswith("_c.h5")]
            
        return 


    #####################################################################################
    def run(self, plot_acc=False):

        ############################### 
        ### Open file to count run time
        if self.reset_backend:
            ftime = open(self.save_dir + "timing_" + self.method + ".txt", "w")
            ftime_0 = ptime.time()
            ftime.write("Started simulation at %s.\n" %
                       (ptime.ctime()))
            ftime.write("with " + self.method + " method.\n")
        else:
            ftime = open(self.save_dir + "timing_" + self.method + ".txt", "a")
            ftime_0 = ptime.time()
            ftime.write('\n')
            ftime.write("Re-started simulation at %s.\n" %
                       (ptime.ctime()))
            ftime.write("with " + self.method + " method.\n")

        ############################### 
        try:
            if self.method == "emcee":
                ### Call emcee with a pool of CPUs
                if self.n_cpus>1:
                    if self.do_mpi:
                        with self.pool:
                            if not self.pool.is_master():
                                self.pool.wait()
                                sys.exit(0)
                            self.do_emcee()
                    else:
                        with self.pool:
                            self.do_emcee()   
                else:
                    self.do_emcee()

            if self.method == "emcee2":
                ### Call emcee with a pool of CPUs
                if self.n_cpus>1:
                    with self.pool:
                        self.do_emcee_mybackend()
                else:
                    self.do_emcee_mybackend()

            elif self.method == "ptmcmc":
                ### Run multiprocess and call Parallel Tempering MCMC
                if self.n_cpus>1:
                    list_of_runs = [i for i in range(self.n_cpus)]
                    with self.pool:
                        self.pool.map(self.do_ptmcmc, list_of_runs)
                    self.pool.close()
                    self.pool.join()
                else:
                    self.do_ptmcmc(1)
        
            ############################### 
            ### Write timing information for successful run 
            ftime.write("Finished on %s \n" %
                       (ptime.ctime()))
            ftime.write("Ran {:d} iterations total \n".format(self.n_iter))
            ftime.write("Used {:d} CPUs \n".format(self.n_cpus))
            ftime.write("Total time of {:.1f} s\n".format(ptime.time()-ftime_0))
            ftime.close()
            print("- SUCCESS -")

            ############################### 
            ### Plot acceptance rate information
            if plot_acc:
                fig, ax = plt.subplots()
                ax.plot(self.acc_list[:,0], label="Walker {:d}".format(0), c='navy')
                if self.n_walk > 3:
                    ax.plot(self.acc_list[:,self.n_walk//2], label="Walker {:d}".format(self.n_walk//2), c='slateblue')
                    ax.plot(self.acc_list[:,self.n_walk-1], label="Walker {:d}".format(self.n_walk-1), c='teal')
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Acceptance rate")
                ax.set_xlim(0,self.sampler.iteration-1)
                ax.set_ylim(0,1)
                ax.legend(edgecolor='none')


        ############################### 
        except Exception as e :
            ### Write timing and error for unsuccesful run 
            print("- FAILED -")
            import traceback
            print(traceback.format_exc())
            # self.pool.close()
            # self.pool.join()
            ftime.write("Failed on %s \n" %
                       (ptime.ctime()))
            ftime.write("Reason : %s \n" % e)
            ftime.write("Total time before failure: {:.1f} s\n".format(ptime.time()-ftime_0))
            ftime.close()

        return 


    #####################################################################################
    def do_emcee(self):    

        ############################### 
        ### Check if starting from a fresh backend 
        if self.reset_backend:
            print('Reset start')
            pos = self.pos[0,:,:]
            ### Separate numpy array to save acceptance fraction
            self.acc_list = np.zeros((self.n_iter,self.n_walk))
        else:
            if not self.allow_warm_start:
                print('Cannot perform warm start since no previous MCMC data found')
                return
            else:
                print('Warm start')
                ### Having pos = None doesn-t work very well.
                pos = self.sampler._previous_state[0] 
                ### Extend list of acceptance fraction so that we add Niter to the last stored value
                nit_sofar = self.sampler.iteration
                ### NOTE:  very inefficient with numpy. Reason why emcee2 was created 
                self.acc_list = np.load(self.save_dir + 'acceptance_rate.npy') 
                self.acc_list.resize((nit_sofar+self.n_iter,self.n_walk)) 
                
        self.run_finished = False
        ######################################################################################### 
        ### Iterate sampler (allows to save acceptance rate along the way)
        for sample in self.sampler.sample(pos, iterations=self.n_iter, progress=self.progress,
                           progress_kwargs={'ncols':80, 'leave':True, 'position':0, 'file':sys.stdout}):
            itt = self.sampler.iteration-1
            acc_yet = self.sampler.acceptance_fraction
            self.acc_list[itt,:] = acc_yet

            #######################################################
            ### Every n_save steps, copy backend in accessible file
            if (itt%self.n_save==0 and itt!=0):
                shutil.copyfile(self.filename, self.save_dir + 'chains_emcee_c.h5')
                np.save(self.save_dir + "acceptance_rate", self.acc_list)

            #######################################################
            ### For simulations on cluster: Do not plot progress bar
            if (itt%200==0 and itt!=0 and not self.progress):
                print("Saving iteration {:d}".format(itt))
                
        #######################################################
        ### Final save of backend and acceptance rate file 
        shutil.copyfile(self.filename, self.save_dir + 'chains_emcee_c.h5')
        np.save(self.save_dir + "acceptance_rate", self.acc_list)
        ### NOTE: emcee2 created to use hdf5 for acceptance rate

        self.run_finished = True
        self.allow_warm_start = True
        return 
    
    
    #####################################################################################
    def do_emcee_mybackend(self):    

        ############################### 
        ### Check if starting from a fresh backend 
        if self.reset_backend:
            print('Reset start')
            pos = self.pos[0,:,:]
            old_accepted = 0 
            
        else:
            if not self.allow_warm_start:
                print('Cannot perform warm start since no previous MCMC data found')
                return
            else:
                print('Warm start')
                ### Read position from existing hdf5 file 
                with h5py.File(self.filename, "r") as f:
                    pos = f["chain"][-1,:,:]  # Shape: (nsteps, nwalkers, ndim)
                    old_accepted = f["accepted"][-1,:]
                
        self.run_finished = False
        ### Separate numpy array to save acceptance fraction
        accepted_buffer = []
        log_prob_buffer = []
        chain_buffer = []
        # blobs_buffer = []
        ######################################################################################### 
        ### Iterate sampler (allows to save acceptance rate along the way)
        for itt, sample in enumerate(self.sampler.sample(pos, iterations=self.n_iter, progress=self.progress,
                           progress_kwargs={'ncols':80, 'leave':True, 'position':0, 'file':sys.stdout})):
            
            ### If the simulation has been restarted, the acceptance rate will be biased
            ### As it does not know how many have been accepted/rejected before. 
            new_accepted = old_accepted + self.sampler.acceptance_fraction.copy()*(itt+1)
            accepted_buffer.append(new_accepted)
            log_prob_buffer.append(sample.log_prob.copy())
            chain_buffer.append(sample[0].copy())
            # blobs_buffer.append(sample.blobs.copy())

            #######################################################
            ### Every n_save steps, copy backend in accessible file
            if (itt%self.n_save==0 and itt!=0):
                
                # Append to HDF5 file and clear buffers
                self._append_backend(np.array(chain_buffer), 
                                    np.array(log_prob_buffer), 
                                    np.array(accepted_buffer))#, 
                                    # np.array(blobs_list) )
                shutil.copyfile(self.filename, self.save_dir + 'chains_emcee_c.h5')

                accepted_buffer = []
                log_prob_buffer = []
                chain_buffer = []
                # blobs_buffer = []

            #######################################################
            ### For simulations on cluster: Do not plot progress bar
            if (itt%200==0 and itt!=0 and not self.progress):
                print("End of iteration {:d}".format(itt))
                
        #######################################################
        ### Final save of backend and acceptance rate file 
        self._append_backend(np.array(chain_buffer), 
                            np.array(log_prob_buffer), 
                            np.array(accepted_buffer))#, 
                            # np.array(blobs_list) )
        shutil.copyfile(self.filename, self.save_dir + 'chains_emcee_c.h5')
        
        self.run_finished = True
        self.allow_warm_start = True
        return


    #####################################################################################
    def do_ptmcmc(self, irun):
        ### Each cpu will run separately in pool
        ### Independant hdf5 files need to be initialized
        self._init_backend(self.n_temp, self.n_dim, overwrite=self.reset_backend, irun=irun) 

        ############################### 
        ### Check if starting from a fresh backend 
        if self.reset_backend:
            print('Reset start')
            pos = self.pos[:,0,:]
            old_accepted = 0 
            
        else:
            if not self.allow_warm_start:
                print('Cannot perform warm start since no previous MCMC data found')
                return
            else:
                print('Warm start')
                ### Read position from existing hdf5 file 
                with h5py.File(self.filename, "r") as f:
                    pos = f["chain"][-1,:,:]  # Shape: (nsteps, ntemp, ndim)
                    old_accepted = f["accepted"][-1,:]
                
        self.run_finished = False
        args_lp = (self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC)


        ### Create empty arrays with space for tempering variables
        chain_buffer    = []#mout = np.zeros((self.n_iter, self.n_temp, self.n_dim))
        log_prob_buffer = []#misfit = np.zeros((self.n_iter,self.n_temp))
        accepted_buffer = []#accepted = np.zeros((self.n_iter,self.n_temp))

        current_model   = np.zeros((self.n_temp,self.n_dim))
        candidate_model = np.zeros((self.n_temp,self.n_dim))
        count_swaps     = []#np.zeros((self.n_iter,self.n_temp-1))
        nacc            = np.zeros(self.n_temp)
        llcurrent       = np.zeros(self.n_temp)

        acceptance_swap = []#np.zeros((self.n_iter,self.n_temp-1))
        list_of_swaps   = []
        saveTladder     = []#np.zeros((self.n_iter,self.n_temp))
        proposal_cov = np.array([np.eye(self.n_dim)*self.stepsz for tc in range(self.n_temp)]) 
        # proposal_cov = np.array([np.eye(self.n_dim) * (2.4**2 / self.n_dim) for tc in range(self.n_temp)])
        cov_hist = proposal_cov.copy()
        mean_hist = pos.copy()
        cov_scale=1
        stp_cold = []

        '''TODO: change stepsize and model generator'''
        
        ### Initialize arrays with initial model
        Tladder = self.Tladder.copy()
        saveTladder.append(self.Tladder)
        acceptance_swap.append(nacc)
        count_swaps.append(nacc)

        ### Calculate initial misfit
        current_model = pos 
        llzero = []
        if self.n_blobs==0:
            for tc in range(self.n_temp):
                llzero.append(log_probability(current_model[tc,:], *args_lp))
        else:
            for tc in range(self.n_temp):
                llzero.append(log_probability(current_model[tc,:], *args_lp)[0])
        log_prob_buffer.append(llzero.copy())
        chain_buffer.append(current_model.copy())
        accepted_buffer.append(nacc)
        

        ##############################################################
        ### MAIN LOOP 
        if self.progress:
            loop_range = tqdm(range(1,self.n_iter),ncols=80, leave=True, position=0, file=sys.stdout)
        else: 
            loop_range = range(1,self.n_iter)
        for k in loop_range:
            
            ##############################################################
            ### ADAPTATIVE TEMPERATURE STEP (TODO: DOESN'T WORK FOR NOW)
            ### Try adaptative temperature scaling (from ptemcee)
            # if self.n_temp>1 and self.do_swap and self.use_adapt:
            #     if k>2:
            #         ratios = acceptance_swap[-1]
            #         # Modulate temperature adjustments with a hyperbolic decay.
            #         decay = self.adapt_tlag / (k + self.adapt_tlag)
            #         kappa = decay / self.adapt_nu
            #         # Construct temperature adjustments.
            #         dSs = kappa * (ratios[:-1] - ratios[1:])
            #         # Compute new ladder (hottest and coldest chains don't move).
            #         deltaTs = np.diff(Tladder[:-1])
            #         print(deltaTs.size, Tladder.size, ratios.size, dSs.size)
            #         deltaTs *= np.exp(dSs)
            #         Tladder[1:-1] = (np.cumsum(deltaTs) + Tladder[0])
            ### Save new temperature ladder
            # saveTladder[k,:] = self.Tladder.copy()


            ##############################################################
            ### METROPOLIS-HASTINGS STEP : Loop on temperatures
            for tc in range(self.n_temp):
                    
                llcurrent[tc] = log_prob_buffer[-1][tc] #misfit[k-1,tc]

                ### Generate a model:
                candidate_model[tc,:] = move_model(current_model[tc,:], proposal_cov[tc])#self.stepsz)
                ### Verify that the proposed model is in the prior bounds. If not, reject it.
                if self.n_blobs ==0:
                    llcandidate = log_probability(candidate_model[tc,:],*args_lp)
                else:
                    llcandidate = log_probability(candidate_model[tc,:],*args_lp)[0]

                if not np.isinf(llcandidate):
                    ### Evaluate the logarithm of the acceptance ratio.
                    ### PT: the exponent (1/Tc) is added via a multiplication in log space
                    logalpha = (llcandidate - llcurrent[tc]) * 1/Tladder[tc]
                    ### Take the minimum of the log(alpha) and 0.
                    logalpha = min(0.,logalpha)
                    ### Generate a U(0,1) random number and take its logarithm.
                    logt = np.log(np.random.rand())
                    # print(llcandidate , llcurrent[tc], logalpha, logt < logalpha)

                    ### Accept or reject the step.
                    if (logt < logalpha):
                        ### Accept the step. Don't forget the deep copy of numpy arrays.
                        current_model[tc,:] = candidate_model[tc,:].copy()
                        nacc[tc] = nacc[tc] + 1
                        llcurrent[tc] = llcandidate
                        # if tc==0:
                        #     print("accepted", llcurrent[tc])
                    else :
                        ### Reject the step.
                        ### Nothing changes
                        # if tc==0:
                        #     print("rejected", llcurrent[tc])
                        pass 

            accepted_buffer.append(nacc.copy())
            # print(accepted_buffer[-1]/k)

            ##############################################################
            ### PARALLEL TEMPERING STEP : DONE EVERY <FREQ_SWAP> ITERATION
            swaps = []
            if self.n_temp>1 and self.do_swap and (k%self.n_swaps == 0):

                ### Loop over temperatures
                newswaps = np.zeros(self.n_temp)
                for tc in range(self.n_temp) :

                    ### Select two distinct random chains
                    # pchain, qchain = np.random.choice(num, size=2, replace = False)
                    pchain, qchain = random.sample(self.numtemp, 2)

                    ### Get current posterior probability of model for each chain
                    logrq = llcurrent[qchain]
                    logrp = llcurrent[pchain]

                    ### Compute acceptance ratio of swap
                    logalpha_T = (1/Tladder[pchain]-1/Tladder[qchain]) * (logrq - logrp)
                    logt_T = np.log(np.random.rand())

                    ### Accept or reject the swap.
                    if logalpha_T > logt_T :
                        newswaps[pchain] = 1 ### Count as one swap 
                        newswaps[qchain] = 1 ### Count as one swap 

                        ### Switch current models, with deepcopy
                        aswitch = current_model[pchain,:].copy()
                        current_model[pchain,:] = current_model[qchain,:]
                        current_model[qchain,:] = aswitch

                        ### Finally, switch likelyhood information too as we
                        ### will go back into the loop to exchange p and q
                        ### again for NTP temperatures
                        llswitch = llcurrent[pchain].copy()
                        llcurrent[pchain] = llcurrent[qchain]
                        llcurrent[qchain] = llswitch

                        ### Finally, save swaps
                        # print("swapped ",pchain, " and ", qchain)
                        swaps.append([pchain, qchain])

                ### Count number of swaps performed for each couple of adjacent chains
                # for a in range(self.n_temp-1):
                #     counts[k,a]= counts[k-1,a] + swaps.count(adjacents_a[a]) + swaps.count(adjacents_b[a])
                count_swaps.append(count_swaps[-1] + newswaps)
                ### Computes acceptance rate of swap of adjacent chains
                acceptance_swap.append( count_swaps[-1]/k * self.n_swaps ) #counts[k,:]/k * nswaps
                # print("Acc swaps ", acceptance_swap[-1])
                # list_of_swaps.append(swaps)

            ##############################################################
            ### Record the result, after the Parallel Tempering step
            ### Deepcopy very important ! 
            chain_buffer.append(current_model.copy()) # mout[k,:,:] = current_model
            log_prob_buffer.append(llcurrent.copy()) # misfit[k,:] = llcurrent


            ##############################################################
            ### ADAPTATIVE METROPOLIS STEP
            if self.use_adapt:
                ### From Pypesto: doesn't work well in combination with Parallel tempering 
                
                decay_constant= 0.51
                threshold_sample = 1 
                reg_factor = 1e-6 
                target_acceptance_rate = 0.234

                update_rate = k ** (-decay_constant)

                for tc in range(self.n_temp):
                    
                    ### compute historical mean and covariance
                    mean_hist[tc,:] = (1 - update_rate) * mean_hist[tc,:] + update_rate * current_model[tc]
                    dx = current_model[tc] - mean_hist[tc,:]
                    cov_hist[tc,:,:] = (1 - update_rate) * cov_hist[tc,:,:] + update_rate * dx.reshape((-1, 1)) @ dx.reshape((1, -1))

                    ### compute covariance scaling factor based on the target acceptance rate
                    cov_scale *= np.exp(  (nacc[tc]/k - target_acceptance_rate) / np.power(k + 1, decay_constant) )

                    ### set proposal covariance
                    proposal_cov[tc,:,:] = cov_scale * cov_hist[tc,:,:]

                    ### Avoid zero on diagonal by regularization: 
                    eig = np.linalg.eigvals(proposal_cov[tc,:,:])
                    eig_min = min(eig)
                    if eig_min <= 0:
                        proposal_cov[tc,:,:] += (abs(eig_min) + reg_factor) * np.eye(self.n_dim) 

                # stp_cold.append(proposal_cov[0,0,0])
                # print(stp_cold[-1])


            ###############################################################
            ### SAVE IN HDF5 FILE 
            if k%self.n_save==0 or k==self.n_iter:
                ### Append buffered items: everything except last one 
                self._append_backend(np.array(chain_buffer[:-1]), 
                                np.array(log_prob_buffer[:-1]), 
                                np.array(accepted_buffer[:-1]))#, 
                                # np.array(blobs_list) )
                shutil.copyfile(self.filename, self.save_dir + 'chains_ptmcmc_{:d}_c.h5'.format(irun))

                ### To save temperature information, we will create a different 
                ### hdf5 file later 
                # save_swaps = np.array(list_of_swaps, dtype=object)
                # np.save(dir_res + "Swaps_process_{:02d}".format(index), save_swaps)
                # np.save(dir_res + "Accswap_process_{:02d}".format(index), acceptance_swap)
                # np.save(dir_res + "Tladder_process_{:02d}".format(index), saveTladder)

                ### Clean buffer: restart from last one which was not saved 
                accepted_buffer = [accepted_buffer[-1]]
                log_prob_buffer = [log_prob_buffer[-1]]
                chain_buffer = [chain_buffer[-1]]

            #######################################################
            ### For simulations on cluster: Do not plot progress bar
            if (k%200==0 and k!=0 and not self.progress):
                print("End of iteration {:d}".format(k))
                    
        ### Save last one 
        self._append_backend(np.array(chain_buffer), 
                                np.array(log_prob_buffer), 
                                np.array(accepted_buffer))#, 
                                # np.array(blobs_list) )
        shutil.copyfile(self.filename, self.save_dir + 'chains_ptmcmc_{:d}_c.h5'.format(irun))

        self.run_finished = True
        self.allow_warm_start = True

        return
    
    
    ##########################################################################################################  
    ### PLOTTING FUNCTIONS
    ########################################################################################################## 
    '''
    Functions to : 
        - load results, 
        - process subsurface models for plotting 
        - Inspect MCMC chains 
        - Inspect autocorrelation time of the chains (check convergence)
        - Display source location on map 
        - Display inverted subsurface model and histograms
    ''' 

    ##########################################################################################################  
    ### INTERPOLATING SUBSURFACE MODELS 
    ##########################################################################################################  
    def _interpolate_solutions(self, depth_interpolation=np.linspace(0., 100., 1000), int_method="1"):

        self.depth_interpolation = depth_interpolation
        dh = np.diff(depth_interpolation)[0]

        N_intp = depth_interpolation.size
        # wr_vs = self.wr_order[3]
        # wr_ps = self.wr_order[4]
        wr_hl = self.wr_order[5]
        N_layer= wr_hl.size

        if int_method == "1":
            ### Smoother transitions at interfaces
            shape = (self.n_end, N_intp)
        elif int_method == "2":
            ### Sharper transitions at interfaces
            shape = (self.n_end, N_intp+N_layer)
            
        self.vs_interpolation = np.zeros(shape)
        self.vp_interpolation = np.zeros(shape)
        self.ps_interpolation = np.zeros(shape)
        self.ids = np.repeat(np.arange(self.n_end), shape[1])

        ############################################################################ 
        if self.blobs is None:
            args = [(i, self.flat_samples[i,:], self.blobs, self.is_inverted, self.wr_order, 
                depth_interpolation, int_method, dh, N_layer, N_intp) for i in range(self.n_end)]
        else:
            args = [(i, self.flat_samples[i,:], self.blobs[i], self.is_inverted, self.wr_order, 
                depth_interpolation, int_method, dh, N_layer, N_intp) for i in range(self.n_end)]
        
        ### Iterate on all selected posterior models
        with Pool(int(cpu_count()/2)) as pool:
            results = list(tqdm(
                pool.starmap(_interpolate_one_model_, args),
                total=self.n_end, ncols=80, leave=True, position=0
            ))
        # pool.close()
        # pool.join()

        for i, vs_interp, vp_interp, ps_interp in results:
            self.vs_interpolation[i, :] = vs_interp
            self.vp_interpolation[i, :] = vp_interp
            self.ps_interpolation[i, :] = ps_interp
        
        self.vs_interpolation = self.vs_interpolation.flatten()
        self.vp_interpolation = self.vp_interpolation.flatten()
        self.ps_interpolation = self.ps_interpolation.flatten()
        self.depth_interpolation_all = np.tile(self.depth_interpolation, self.n_end)
    
        return()


    ##########################################################################################################  
    ### PROCESSING SOLUTION (DOWNSAMPLING)
    ##########################################################################################################  
    def process_solutions(self, discard=0, thin=1, crit=-np.inf):

        print("Loading samples...")
        self.solutions_postprocessed = False

        opt = dict(discard=discard, thin=thin, flat=True)

        ### Flatten all the walkers into one chain (doesn't keep the time evolution of the McMC chain)
        if self.method == "emcee":
            self.flat_samples = self.sampler.get_chain(**opt)
            self.log_flat_samples = self.sampler.get_log_prob(**opt)
            self.blobs = self.sampler.get_blobs(**opt)

        else:   
            with h5py.File(self.filename, "r") as f:
                chain = f["chain"][:]  # Shape: (nsteps, nwalkers, ndim)
                log_prob = f["log_prob"][:]  # Shape: (nsteps, nwalkers)
                # acceptance_rate = f["acceptance_rate"][:]  # Shape: (nsteps,nwalkers)
                self.flat_samples = self._get_value(chain, **opt)
                self.log_flat_samples = self._get_value(log_prob, **opt)

        ### Remove models that are outside a certain criterion, if any. 
        self.flat_samples = self.flat_samples[np.where(self.log_flat_samples > crit)]
        if self.blobs is not None:
            self.blobs = self.blobs[np.where(self.log_flat_samples > crit)]
        self.log_flat_samples = self.log_flat_samples[np.where(self.log_flat_samples > crit)]

        ### Number of remaining models for statistical analysis
        self.n_end = self.log_flat_samples.size
        print("Extracted {:d} subsurface models".format(self.n_end))

        self.solutions_postprocessed = True
        return()


    ##########################################################################################################  
    ### CALCULATING THE MAP FROM KERNEL DENSITY ESTIMATES OR OTHER METHODS 
    ##########################################################################################################  
    def get_MAP(self, fac_nm = 5, method ="Meanshift", NM=None, do_test=False, ncomp = None):

        ### We use a seed to ensure results don't change between two plots
        ### We select models randomly in the distribution to evaluate the MAP 
        random_state = np.random.RandomState(seed=21549)

        self.ts_map  = [] 
        self.lat_map = [] 
        self.lon_map = [] 
        self.hs_map = [] 
        self.vs_map = []  
        self.vp_map = [] 
        self.ps_map = []  
        self.velocity_model_map = []

        def find_max_approx(gmm):
            dens = [gmm.score_samples([mean])[0] for mean in gmm.means_]
            densest_mod = gmm.means_[np.argmax(dens)]
            return(densest_mod)

        ##########################################¨
        if method == "Meanshift":
            from sklearn.cluster import MeanShift, estimate_bandwidth

            ### Downsample models (starts to become really slow with >2e4)
            # if NM is None: 
            #     NM = int(min(1e4, self.n_end//fac_nm))
            # NM=int(2e4)
            print("Finding MAP using {:d} iterations".format(NM))
            id_models = random_state.choice(self.flat_samples.shape[0], size=NM, replace=False)
            samples_models = self.flat_samples[id_models,:]

            ### Scale model points: 
            samples_models_r = samples_models.copy()
            scaled_models = samples_models_r.copy()
            ### Scale to [0,1]
            minmax = np.zeros((samples_models_r.shape[1],2))
            for dd in range(samples_models_r.shape[1]):
                minmax[dd,:] = [samples_models_r[:,dd].min(),samples_models_r[:,dd].max()]
                scaled_models[:,dd] = (scaled_models[:,dd]-minmax[dd,0])/(minmax[dd,1]-minmax[dd,0])
            self.map_estimates = []

            ### Mean Shift Approach
            bandwidth = estimate_bandwidth(scaled_models, quantile=0.2, n_samples=NM)  # Adjust quantile
            print("estimated bandwidth: ", bandwidth)
            mean_shift = MeanShift(bandwidth=bandwidth, n_jobs=10)#, bin_seeding=True)
            # mean_shift = MeanShift(bandwidth=0.95*bandwidth, n_jobs=10)#, bin_seeding=True)
            mean_shift.fit(scaled_models)
            print("fit meanshift")
            
            # Find the highest-density cluster
            for i, cc in enumerate(mean_shift.cluster_centers_):
                cc = cc*(minmax[:,1]-minmax[:,0]) + minmax[:,0]
                ### normal 
                self.map_estimates.append(cc)

            densest_cluster = np.argmax(np.bincount(mean_shift.labels_))
            mode_mean_shift = mean_shift.cluster_centers_[densest_cluster]
            # print("Mean Shift Estimated Mode:", mode_mean_shift)
            mode_mean_shift = mode_mean_shift * (minmax[:,1]-minmax[:,0]) + minmax[:,0]
            self.map_estimate = mode_mean_shift
            print("Mean Shift Estimated Mode:", self.map_estimate)

        ##########################################
        ### FIND MAP USING GAUSSIAN MIXTURE MODELS
        elif method=="gmm": 
            
            from sklearn.mixture import GaussianMixture
            
            ### Downsample models (starts to become really slow with >1e5)
            if NM is None: 
                NM = int(min(1e4, self.n_end//fac_nm))
            print("Finding MAP using {:d} iterations".format(NM))
            id_models = random_state.choice(self.flat_samples.shape[0], size=NM, replace=False)
            samples_models = self.flat_samples[id_models,:]

            ### Scale model points: 
            scaled_models = samples_models.copy()
            minmax = np.zeros((samples_models.shape[1],2))
            for dd in range(samples_models.shape[1]):
                minmax[dd,:] = [samples_models[:,dd].min(),samples_models[:,dd].max()]
                scaled_models[:,dd] = (scaled_models[:,dd]-minmax[dd,0])/(minmax[dd,1]-minmax[dd,0])

            ### Evaluate BIC and AIC metrics depending on number of components 
            best_gmm = None
            best_bic = np.inf
            bics = []  ### Bayes information criterion
            aics = []  ### Akaike information criterion
            gmms = []  ### To check result from different number of components 
            ### TOO HIGH NUMBER OF GAUSSIANS GIVE POOR RESULTS 
            if do_test or ncomp is None:
                test_components = list(range(1,13))
            else:
                test_components = [ncomp]
            best_n_components = test_components[0]
        
            self.map_estimates = []
            for icomp, n_components in enumerate(test_components): 
                print(n_components)
                gmmfunc = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
                gmm = gmmfunc.fit(scaled_models)
                bics.append(gmm.bic(scaled_models))
                aics.append(gmm.aic(scaled_models))
                gmms.append(gmm)

                if do_test:
                    def find_max_smooth(data, gmm, lower_bounds=None, upper_bounds=None):
                        if lower_bounds is None:
                            lower_bounds = data.min(axis=0)
                        if upper_bounds is None:
                            upper_bounds = data.max(axis=0)
                        bounds = list(zip(lower_bounds, upper_bounds))  # Convert to [(min, max), ...]

                        def neg_density_func(x,gmm):
                            return(-gmm.score_samples([x])[0])

                        ### DIFFERENTIAL EVOLUTION STEP 
                        # from scipy.optimize import differential_evolution
                        # result_de = differential_evolution(func=neg_density_func, 
                        #                                 bounds=bounds, args=(gmm,),
                        #                                 strategy="best1bin", popsize=20, 
                        #                                 tol=1e-6, maxiter=200)
                        result_approx = find_max_approx(gmm)

                        ### CMA STEP 
                        import cma
                        x0_approx=np.array([25, -7, 124, 10, 
                                            1, 2, 4, 4, 4, 6, 6, 
                                            0.22, 0.22, 0.22, 0.25, 0.25, 0.25,0.25,
                                            2, 3, 5, 20, 370, 130])
                        x0_approx = (x0_approx - minmax[:,0])/(minmax[:,1]-minmax[:,0])
                        sigma0 = np.std(data, axis=0).mean()  # Step size based on data spread
                        es = cma.CMAEvolutionStrategy(x0_approx, sigma0, 
                                                    {"popsize": 40, "AdaptSigma": True})
                        es.optimize(neg_density_func, args=(gmm,), iterations=400)
                        result_cma = es.result.xbest
                        #best_density = -neg_gmm_density(result_cma)

                        # x0 = result.x  # Best solution found by DE
                        return(result_approx, result_cma)
                
                    # x0, x0b = find_max_smooth(scaled_models, gmm)#,min, max)
                    # print("DE: ", x0*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                    # print("Density : ", gmm.score_samples([x0])[0])
                    # print("CMA: ", x0b*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                    # print("Density : ", gmm.score_samples([x0b])[0])
                    # self.map_estimates.append(x0b*(minmax[:,1]-minmax[:,0]) + minmax[:,0])

                    x0_approx = find_max_approx(gmm)
                    # print("Approx: ", x0_approx*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                    # print("Density : ", gmm.score_samples([x0_approx])[0])
                    self.map_estimates.append(x0_approx*(minmax[:,1]-minmax[:,0]) + minmax[:,0])

                else:
                    densest_region = find_max_approx(gmm)
                    self.map_estimates.append(densest_region*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                
                if bics[-1] < best_bic:
                    best_bic = bics[-1]
                    best_gmm = gmm
                    best_n_components = n_components


            ### Check BIC AND AIC 
            # if do_test:
            #     print("Best components : ", best_n_components)
            #     dens = [] 
            #     for it in range(len(test_components)): 
            #         dens.append(max([best_gmm.score_samples([mean])[0] for mean in gmms[it].means_]))
            #     fig, ax = plt.subplots(figsize=(5,4))
            #     ax.plot(test_components, bics, marker="o", label="BIC")
            #     ax.plot(test_components, aics, marker="s", label="AIC")
            #     axb = ax.twinx()
            #     axb.plot(test_components, dens, marker="^", c="k") 
            #     axb.plot()
            #     ax.set_xlabel("Number of components")
            #     ax.set_ylabel("Bayesian criteria")
            #     axb.set_ylabel("Max density at Gaussian center")
            #     ax.legend(frameon=False)
            #     ax.set_xlim(0.5,max(test_components)+0.5)
            #     fig.tight_layout()
                

            ### Extracting the densest point is not enough: The N-dimensional space is too sparse !
            # density = best_gmm.score_samples(scaled_models.T)
            # max_density_idx = np.argmax(density)
            # self.map_estimate = samples_kde[:,max_density_idx]
            # print("Coordinates of max proba points:", self.map_estimate)

            ### Retrieve the center of the densest Gaussian instead
            max_density = -np.inf
            densest_region = None
            covar = None
            densest_region = find_max_approx(best_gmm)
            max_density = best_gmm.score_samples([densest_region])[0]
            self.map_estimate = densest_region*(minmax[:,1]-minmax[:,0]) + minmax[:,0]

            ### Rescale covariance: 
            scaling = np.eye(self.n_dim) * (minmax[:,1]-minmax[:,0])
            for u in range(best_gmm.means_.shape[0]):
                best_gmm.covariances_[u,:,:] = np.dot(scaling, np.dot(best_gmm.covariances_[u,:,:], scaling) )
            self.gmm_covariance = best_gmm.covariances_
            self.gmm_weights = best_gmm.weights_
            # print("Center of densest Gaussian: ", densest_region)

            if do_test:
                plt.show()


        ###########################################
        ### FIND MAP USING KERNEL DENSITY ESTIMATES 
        ### NOTE: Despite using many sample, the KDE is still very unstable
        elif method=="kde":

            NM = int(min(1e4, self.n_end//fac_nm))
            print("Finding MAP using {:d} iterations".format(NM))
            id_models = random_state.choice(self.flat_samples.shape[0], size=NM, replace=False)
            samples_models = self.flat_samples[id_models,:]

            used_kde = "statsmodel"
            # used_kde="scipy"
            # used_kde="sklearn"

            ### using Scipy method 
            if used_kde == "scipy":
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(samples_models.T)#, bw_method=1.)
                ## Evaluate the density at each sample point
                densities_kde = kde(samples_models.T)


            ### Using statsmodel to find best solution:
            elif used_kde=="statsmodel":
                var_type = 'c'
                for i in range(1, self.n_dim):
                    var_type += 'c'

                ### Scale model points: 
                scaled_models = samples_models.copy()
                minmax = np.zeros((samples_models.shape[1],2))
                for dd in range(samples_models.shape[1]):
                    minmax[dd,:] = [samples_models[:,dd].min(),samples_models[:,dd].max()]
                    scaled_models[:,dd] = (scaled_models[:,dd]-minmax[dd,0])/(minmax[dd,1]-minmax[dd,0])

                ### Do not scale 
                # scaled_models = samples_models.copy()
                # minmax = np.zeros((samples_models.shape[1],2))
                # for dd in range(samples_models.shape[1]):
                #     minmax[dd,:] = [0.,1.]
                    

                import statsmodels.api as sm 
                kde = sm.nonparametric.KDEMultivariate(data=scaled_models, var_type=var_type, bw='normal_reference')
                ### Cross-validation is impossible because too slow
                # bandwidths = kde.bw ### The bandwidth found by statsmodel
                ### Evaluate the density at each sample point
                # densities_kde = kde.pdf(samples_models)
                # print(np.log(kde.pdf(scaled_models[0,:])))
                
                
                def find_max_smooth(data, gmm, lower_bounds=None, upper_bounds=None):
                    if lower_bounds is None:
                        lower_bounds = data.min(axis=0)
                    if upper_bounds is None:
                        upper_bounds = data.max(axis=0)
                    bounds = list(zip(lower_bounds, upper_bounds))  # Convert to [(min, max), ...]

                    def neg_density_func(x,kde):
                        return(-np.log(kde.pdf(x)))

                    ### DIFFERENTIAL EVOLUTION STEP 
                    # print("starting DE, t= ", ptime.time())
                    # from scipy.optimize import differential_evolution
                    # result_de = differential_evolution(func=neg_density_func, 
                    #                                 bounds=bounds, args=(kde,),
                    #                                 strategy="best1bin", popsize=20, 
                    #                                 #tol=1e-6,
                    #                                 disp=True, 
                    #                                 maxiter=200)

                    ### SAMPLE SEARCH STEP 
                    # x0_approx = scaled_models[np.argmax(kde.pdf(scaled_models))]
                    x0_approx=np.array([25, -7, 124, 10, 1,2,4,4,4,6,6, 0.22, 0.22, 0.22, 0.25, 0.25, 0.25,0.25,
                                2,3,5,20,370,130])
                    # x0_approx = (x0_approx - minmax[:,0])/(minmax[:,1]-minmax[:,0])
                    
                    # print("Approx: ", x0_approx*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                    print("Density : ", np.log(kde.pdf(x0_approx)) )

                    ### CMA STEP 
                    import cma
                    print("starting CMA, t= ", ptime.time())
                    sigma0 = 1.#np.std(data, axis=0).mean()  # Step size based on data spread
                    es = cma.CMAEvolutionStrategy(x0_approx, sigma0, 
                                                {"popsize": 40, "AdaptSigma": True})
                    es.optimize(neg_density_func, args=(kde,), iterations=200)
                    result_cma = es.result.xbest
                    #best_density = -neg_gmm_density(result_cma)

                    # x0 = result.x  # Best solution found by DE
                    # return(result_de.x, result_cma)
                    return(x0_approx, result_cma)
                
                x0, x0b = find_max_smooth(scaled_models, kde)#,min, max)
                print("DE: ", x0*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                print("Density : ", np.log(kde.pdf(x0)) ) 
                print("CMA: ", x0b*(minmax[:,1]-minmax[:,0]) + minmax[:,0])
                print("Density : ", np.log(kde.pdf(x0b)))

               
                quit()

            
            ### Scikit-learn method: Using grid search to find best bandwidth. 
            ### Problem: No way to vary the bandwidth according to covariance of data
            ### Not possible to get characteristics of the kernel 
            elif used_kde == "sklearn":
                from sklearn.model_selection import GridSearchCV
                from sklearn.neighbors import KernelDensity

                params = {'bandwidth': np.logspace(-1, 1, 20)}
                kdegrid = GridSearchCV(KernelDensity(kernel = 'gaussian'),params)
                kdegrid.fit(samples_models)

                best_bw = kdegrid.best_params_['bandwidth']
                print("best bandwidth: {0}".format(best_bw))

                kde = kdegrid.best_etimator_
                densities_kde = kde.score_samples(samples_kde)


            ### Find model with highest value for density: 
            max_density_idx = np.argmax(densities_kde)
            self.map_estimate = samples_models[:,max_density_idx]


        ######################################
        ### EXTRACT USEFUL MAP MODEL VARIABLES
        for i in range(len(self.map_estimates)): 
            self.ts_map.append( self.map_estimates[i][self.wr_order[0]])
            self.lat_map.append(self.map_estimates[i][self.wr_order[1][0]])
            self.lon_map.append(self.map_estimates[i][self.wr_order[1][1]])
            self.hs_map.append( self.map_estimates[i][self.wr_order[2]])
            ### Subsurface models interpolation:
            if self.blobs is None:
                layer_vs_map = self.map_estimates[i][self.wr_order[3]]
                layer_poisson_map = self.map_estimates[i][self.wr_order[4]]
                layer_h_layer_map = self.map_estimates[i][self.wr_order[5]]
            else:
                layer_vs_map, layer_poisson_map, layer_h_layer_map = get_model_from_blob(self.map_estimates[i], self.blobs[0], self.is_inverted, self.wr_order)
            self.velocity_model_map.append(create_model(layer_vs_map, layer_poisson_map, layer_h_layer_map))

        print("MAP model: [" + ", ".join("{:.4g}".format(l) for l in self.map_estimate) + "]")

        return()


    ##########################################################################################################  
    ### TRANSFORMING STATISTIC OF LAYER THICKNESSES INTO STATISTICS OF INTERFACE DEPTH 
    ##########################################################################################################  
    def make_hist_depth(self, samples, blobs, map_estimate, do_MAP = True, NPrior = 5000000):

        wr_hl = self.wr_order[5]
        wrall_hl = self.wrall_order[5]
        
        samples_depthlayer = samples.copy()
        map_model_depth = map_estimate.copy()
        limsc_depth = [list(i) for i in self.prior_inv]

        ### Counter for non-inverted variables 
        cvar = 0
        for iv in self.is_inverted[:5]:
            if len(iv)>1:
                for j in range(len(iv)):
                    cvar += np.size(iv[j]) - np.sum(iv[j])
            else:
                cvar += len(iv) - sum(iv)
        ### Counter for inverted variables 
        cinv = 0
                    
        ### Counter for depth 
        depth_sum = 0
        depth_map_sum = 0

        ### Loop on both inverted and non-inverted parameters to find depth 
        for al, is_inv in enumerate(self.is_inverted[5]):
            if is_inv:
                depth_sum = depth_sum + samples[:,wr_hl[cinv]]
                samples_depthlayer[:,wr_hl[cinv]] = depth_sum
                ###
                if do_MAP:
                    depth_map_sum = depth_map_sum + map_model_depth[wr_hl[cinv]]
                    map_model_depth[wrall_hl[cinv]] = depth_map_sum
                cinv += 1
            else:
                depth_sum = depth_sum + blobs["fixed_var_{:d}".format(cvar)]
                if do_MAP:
                    depth_map_sum = depth_map_sum + blobs["fixed_var_{:d}".format(cvar)]
                cvar+=1   
        for al in range(1,len(wrall_hl)):
            ### Redifine prior limits as interface limits 
            limsc_depth[wrall_hl[al]][0] = limsc_depth[wrall_hl[al]-1][0] + limsc_depth[wrall_hl[al]][0]
            limsc_depth[wrall_hl[al]][1] = limsc_depth[wrall_hl[al]-1][1] + limsc_depth[wrall_hl[al]][1]

        ################################
        ### Get overall prior probability                    
        priors_at_depth =  np.zeros((NPrior,wr_hl.size))           
        for al in range(len(self.prior_min_sorted[5])):
            ### Only count places where there is a real uniform law, 
            ### Otherwise it skews the prior distribution towards fixed values
            if self.prior_min_sorted[5][al] != self.prior_max_sorted[5][al]:
                ### Calculate prior distribution of thickness
                if al == 0: 
                    priors_at_depth[:,al]= np.random.uniform(low=self.prior_min_sorted[5][al], high= self.prior_max_sorted[5][al], size=NPrior)
                else:
                    priors_at_depth[:,al] = priors_at_depth[:,al-1] + np.random.uniform(low=self.prior_min_sorted[5][al], high= self.prior_max_sorted[5][al], size=NPrior)
        
        ################################
        ### Look at each summed variable independantly
        ### NOTE: This is a more exact way of comparing posterior interfaces to "pure chance"
        ### It determine if interface N has posterior probability superior to 
        ### Posterior distribution of interface N-1 + a uniform prior distribution of layer thickness N 
        ### Instead of just looking at 
        ### Prior distribution of interface N-1 + prior distribution of layer thickness N 
        mixed_priors = np.zeros((samples_depthlayer.shape[0], wr_hl.size)) 
        for w, iw in enumerate(wr_hl):
            interface_posterior = samples_depthlayer[:,iw]
            if w==0:
                mixed_priors[:,w] = np.random.uniform(low=self.prior_min_sorted[5][w], high= self.prior_max_sorted[5][w], size=interface_posterior.size)
            else:
                ### The posterior of previous interfaces + the prior of new thickness 
                mixed_priors[:,w] = samples_depthlayer[:,iw-1]  + np.random.uniform(low=self.prior_min_sorted[5][w], high= self.prior_max_sorted[5][w], size=interface_posterior.size)
        
        return(samples_depthlayer, priors_at_depth, mixed_priors, limsc_depth, map_model_depth)



    ##########################################################################################################  
    ### COLOR PALETTES FOR PLOTS 
    ##########################################################################################################  
    def color_palette(self, color_code=1):
        ### COLOR PROFILE FOR FIGURE 

        ######################################################################################################
        ### COLOR PACKAGE 1: 
        if color_code==1:
            ###For histograms: 
            self.hist_col, self.true_col, self.max_col, self.map_col, self.hist_alpha = "#df2d2d", "navy", "crimson", "#00A272", 0.4
            ### For density plots 
            self.true_coldens, self.max_coldens, self.map_coldens, self.bgcol = "#6a7bff", "#ff1818", "#00d294", "#7b7892"  #"#58adb2"]# #6170e1  #"#4bb89a"
            ### Color map 
            self.dens_cmap = sns.color_palette("flare", as_cmap=True)
            self.loc_cmap = self.dens_cmap

        #######################################################################################################
        ### COLOR PACKAGE original  
        elif color_code==0:
            ###For histograms: 
            self.hist_col, self.true_col, self.max_col, self.map_col, self.hist_alpha = "crimson", "navy", "crimson", "#00A272", 0.4
            ### For density plots  
            self.true_coldens, self.max_coldens, self.map_coldens, self.bgcol = "#006dd8", "#fa6ad8", ["#00d294"], "#6d656e"    #mediumslateblue  #ff1818  "#6328ee"  "#f20694" #44bd3d
            ### Color map 
            self.dens_cmap = sns.color_palette("rocket", as_cmap=True)
            self.loc_cmap = sns.color_palette("rocket_r", as_cmap=True)  

        return()      


    ##########################################################################################################  
    ### VISUALIZE SOLUTION 
    ##########################################################################################################  
    def visualize_solution(self, figsize=(10,5), depth_interpolation=np.linspace(0., 150., 201), Nmod= 100, 
                            bins_hist=20, bins_wave=40, hspace=2.5, wspace=0.1, do_save=False, preserve_aspect=False, 
                            do_MAP=True, do_ML=False, do_truth=True,
                            color_code=1
    ):


        if not hasattr(self, 'solutions_postprocessed'):
            raise TypeError('Please postprocess posterior models with .process_solutions()')
        if not hasattr(self, 'vs_interpolation'):
            ### Recreate finely-sampled subsurface models
            print("Interpolating models...")
            self._interpolate_solutions(depth_interpolation=depth_interpolation)
        elif depth_interpolation[-1] != np.max(self.depth_interpolation_all):
            ### Recreate finely-sampled subsurface models
            print("Interpolating models...")
            self._interpolate_solutions(depth_interpolation=depth_interpolation)

        ####################################################################################################
        ### Get Model information 
        wr_ts = self.wr_order[0]
        wr_Ls = self.wr_order[1]
        wr_hs = self.wr_order[2]
        wr_vs = self.wr_order[3]
        wr_ps = self.wr_order[4]
        wr_hl = self.wr_order[5]

        Nwaves =  len(self.DATA.traces)
        truth_location = self.DATA.truth_location
        truth_velocity = self.DATA.truth_velocity
        do_S, do_P = False, False 

        ####################################################################################################
        ### Load true solution
        periods = self.DATA.target_periods
        t0_true, lat_true, lon_true, source_depth_true = truth_location[0:4]
        # _, vg0_true = compute_vg_n_layers(self.DATA.periods, truth_velocity, max_mode = 1,)
        vs_true_interpolation = build_model_from_thicknesses(truth_velocity, self.depth_interpolation, 2)
        vp_true_interpolation = build_model_from_thicknesses(truth_velocity, self.depth_interpolation, 1)
        ps_true_interpolation = build_model_from_thicknesses(truth_velocity, self.depth_interpolation, 3)

        ####################################################################################################
        ### Load MAP 
        if do_MAP:
            if not hasattr(self, 'map_estimate'):
                self.get_MAP(NM=min(self.n_end, int(2e4)))  
            Nmap = len(self.ts_map)
            self.vs_map = [build_model_from_thicknesses(self.velocity_model_map[i], self.depth_interpolation, 2) for i in range(Nmap)]
            self.vp_map = [build_model_from_thicknesses(self.velocity_model_map[i], self.depth_interpolation, 1) for i in range(Nmap)]
            self.ps_map = [build_model_from_thicknesses(self.velocity_model_map[i], self.depth_interpolation, 3) for i in range(Nmap)]
            ### Calculate vg 
            _, vg_map = compute_vg_n_layers(periods, self.velocity_model_map[0], max_mode = 1)


        ####################################################################################################
        ### Find best solution
        idx_best = np.argmax(self.log_flat_samples)
        profile_vs = self.vs_interpolation[self.ids==idx_best]
        profile_vp = self.vp_interpolation[self.ids==idx_best]
        profile_ps = self.ps_interpolation[self.ids==idx_best]
        depth_best = self.depth_interpolation_all[self.ids==idx_best]

        best_likelihood = log_probability(self.flat_samples[idx_best,:], 
                                        self.prior_lims, self.prior_inv, self.is_inverted, self.wr_order, self.args_MCMC)
        print("Likelihood of best: ", best_likelihood[0] if isinstance(best_likelihood, (tuple, list)) else best_likelihood)

        ####################################################################################################
        ### Pick Nmod best models: 
        #chosen_id = np.argsort(self.log_flat_samples)[-Nmod-1:-1]
        ### Pick Nmod random models to display group velocity curves.
        chosen_id = np.random.choice(self.n_end, size=Nmod, replace=False)
        ### Add the id of the best model :
        chosen_id = np.append(chosen_id, idx_best)
        chosen_vgs = np.zeros((Nmod+1, periods.size))
        chosen_mis = np.zeros(Nmod+1)
        if np.any( [self.DATA.data_vector[i]["S_arrival"] for i in range(Nwaves)] is not None):
            chosen_tS = np.zeros((Nwaves,Nmod+1))
            do_S = True 
        if np.any( [self.DATA.data_vector[i]["P_arrival"] for i in range(Nwaves)] is not None):
            chosen_tP = np.zeros((Nwaves,Nmod+1))
            do_P = True 
        print("Calculating group velocity and arrival times...")
        for ic, idx in enumerate(tqdm(chosen_id, ncols=80, leave=True, position=0, file=sys.stdout)):
            if self.blobs is None:
                ### For backward compatibility
                chosen_vs = self.flat_samples[idx,wr_vs]
                chosen_poisson = self.flat_samples[idx,wr_ps]
                chosen_h_layer = self.flat_samples[idx,wr_hl]
            else:
                ### Convert to the full model including variables not inverted
                chosen_vs, chosen_poisson, chosen_h_layer = get_model_from_blob(self.flat_samples[idx,:], self.blobs[idx], self.is_inverted, self.wr_order)
            ### Make model 
            chosen_velocity_model = create_model(chosen_vs, chosen_poisson, chosen_h_layer)

            ### Calculate vg 
            _, chosen_vg = compute_vg_n_layers(periods, chosen_velocity_model, max_mode = 1)
            if chosen_vg is not None:
                u = chosen_vg[0].size 
                chosen_vgs[ic,:u] = chosen_vg[0]
            chosen_mis[ic] = self.log_flat_samples[idx]

            ### Choose a source position  
            if self.blobs is None:
                chosen_ts = self.flat_samples[idx,wr_ts]
                chosen_lat, chosen_lon = self.flat_samples[idx,wr_Ls[0]], self.flat_samples[idx,wr_Ls[1]]
                chosen_source_depth = self.flat_samples[idx,wr_hs]
            else:
                chosen_ts, chosen_lat, chosen_lon, chosen_source_depth, _, _, _, _ = get_model_from_inverted(self.flat_samples[idx], self.wr_order, self.prior_lims, self.is_inverted)

            ### Calculate ts, tg  
            if do_S or do_P:
                for il, (sta_lat, sta_lon) in enumerate(zip(self.DATA.sta_lats, self.DATA.sta_lons)):

                    sta_dist = gps2dist_azimuth(sta_lat, sta_lon, chosen_lat, chosen_lon)[0]/1e3
                     
                    if self.DATA.data_vector[il]["S_arrival"] is not None:
                        tS = compute_travel_time_SP(  sta_dist, chosen_source_depth, chosen_velocity_model, phase='s' )
                        ### We convert the travel time to an arrival time: 
                        chosen_tS[il,ic] = tS + chosen_ts + self.DATA.tprops_air[il] 

                    if self.DATA.data_vector[il]["P_arrival"] is not None:
                        tP = compute_travel_time_SP(  sta_dist, chosen_source_depth, chosen_velocity_model, phase='p' ) 
                        ### We convert the travel time to an arrival time: 
                        chosen_tP[il,ic] = tP + chosen_ts + self.DATA.tprops_air[il] 
                
                    #print("Calculated tS : ", tS)
                    #print("Calculated tP : ", tP)

        ### Sort by increasing misfit
        misinds = chosen_mis.argsort()
        chosen_vgs = chosen_vgs[misinds,:]
        chosen_mis = chosen_mis[misinds]
        chosen_tS = chosen_tS[:,misinds]


        ####################################################################################################
        ### Get source information 
        if self.is_inverted[0]: 
            t0s = self.flat_samples[:,wr_ts]
        else: 
            t0s = np.random.uniform(self.prior_lims[0][0], self.prior_lims[0][1], size=self.n_end)
        if self.is_inverted[1][0]:
            source_lats = self.flat_samples[:,wr_Ls[0]]
        else:
            source_lats  = np.random.uniform(self.prior_lims[1][0], self.prior_lims[1][1], size=self.n_end)
        if self.is_inverted[1][1]:
            source_lons = self.flat_samples[:,wr_Ls[1]]
        else:
            source_lons  = np.random.uniform(self.prior_lims[2][0], self.prior_lims[2][1], size=self.n_end)
        if self.is_inverted[2]:
            source_depths = self.flat_samples[:,wr_hs]
        else:
            source_depths  = np.random.uniform(self.prior_lims[3][0], self.prior_lims[3][1], size=self.n_end)


        ### COLOR PROFILE FOR FIGURE 
        #############################################################################################################
        if not hasattr(self, "true_col"):
            self.color_palette(color_code=color_code)
        
        #############################################################################################################
        ####################################################################################################
        ### MAKE PLOT 
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=0.07, bottom=0.12, right=0.95, top=0.95)
        ### changes the fontsize of matplotlib, not just a single figure
        fontsize=8/10*figsize[0]
        rcParams.update({'font.size': fontsize})
        ### Axis limits for vp and vs profiles 
        vs_min, vs_max = min([self.prior_inv[i][0] for i in wr_vs]), min( max([self.prior_inv[i][1] for i in wr_vs]) , 1.2*self.vs_interpolation.max())
        ps_min, ps_max = min([self.prior_inv[i][0] for i in wr_ps]), max([self.prior_inv[i][1] for i in wr_ps])
        vp_min, vp_max = vp_from_vs(vs_min, ps_min), min(vp_from_vs(vs_max, ps_max), 1.2*self.vp_interpolation.max())
        if preserve_aspect:
            aspect_ratio = (vp_max-vp_min)/(vs_max-vs_min)
            # grid = fig.add_gridspec(6*Nwaves+2, 5, width_ratios=[1,aspect_ratio,1,1,1])
            grid = fig.add_gridspec(6*Nwaves+2*Nwaves, 5, width_ratios=[1,aspect_ratio,1,1,1])
        else:
            grid = fig.add_gridspec(6*Nwaves+2, 5)
            aspect_ratio = None      
        ### Axes for seismic model results 
        ax_vs = fig.add_subplot(grid[:,0])
        ax_vp = fig.add_subplot(grid[:,1], sharey=ax_vs)
        ### Axes for source and signal results
        ax_t0 = fig.add_subplot(grid[:2*Nwaves, 2])
        ax_dist = fig.add_subplot(grid[3*Nwaves:5*Nwaves, 2])
        ax_depth = fig.add_subplot(grid[6*Nwaves:, 2])
        ax_disp = fig.add_subplot(grid[:3*Nwaves,-2:])
        ax_waves = [] 
        for iwave in range(Nwaves):
            ax_waves.append(fig.add_subplot(grid[5*Nwaves+iwave*3 :5*Nwaves+3*(1+iwave),3:]))
        axs_oth = [ax_t0,ax_dist,ax_depth, ax_disp] + ax_waves
        depth_max = 0.999*self.depth_interpolation.max()  


        ####################################################################################################
        ### PLOT AND SAVE POISSON RATIO IN SEPARATE PLOT 
        fig1 = plt.figure(figsize=(figsize[0]/5,figsize[1]))
        fig1.subplots_adjust(hspace=hspace, wspace=wspace, left=0.3, bottom=0.12, right=0.95, top=0.95)
        ax_ps = fig1.add_subplot(111)
        dminps = 2

        hvac, valminps, valmaxps = plot_subsurface(ax_ps, self.ps_interpolation, self.depth_interpolation_all, self.depth_interpolation.size, 
                                                            ps_min, ps_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminps)
        if do_truth:
            ax_ps.plot(ps_true_interpolation, self.depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10, label="True")
        if do_ML:
            ax_ps.plot(profile_ps, depth_best, color=self.max_coldens, linewidth=2, zorder=1, label="Best")
        if do_MAP:
            if Nmap>1:
                cmap = plt.get_cmap("Blues")
                for i in range(len(self.vs_map)):
                    # mapcol = colorskde[i]
                    ax_ps.plot(self.ps_map[i], self.depth_interpolation, color=cmap(i/Nmap), linewidth=2, ls="-.", zorder=100, label="MAP")
            else:
                ax_ps.plot(self.ps_map[0], self.depth_interpolation, color=self.map_coldens, linewidth=2, ls="-.", zorder=100, label="MAP")
        
        
        ### DECOR 
        ax_ps.set_ylim(0, depth_max)
        ax_ps.set_xlim(ps_min,ps_max)
        ax_ps.set_xlabel("Poisson's ratio")
        ax_ps.set_ylabel(r'Depth / [$km$]')
        ax_ps.legend(loc=4, framealpha=0.3, facecolor="k", edgecolor='None', labelcolor="w")
        ax_ps.invert_yaxis()
        
        ### COLORBAR PS 
        cbaxes1 = inset_axes(ax_ps, width="8%", height="30%", loc=3) 
        # cbps = fig.colorbar(hvp, cax=cbaxes, orientation='vertical', ticks=[0, vmaxp] )
        cbps = fig.colorbar(hvac, cax=cbaxes1, orientation='vertical', ticks=[valminps, valmaxps] )
        cbps.ax.set_yticklabels(['0', '{:d}'.format(dminps)])
        cbps.set_label("Model/Mean Density", color="w")
        # cbps.set_label("Probability", color="w")
        cbps.ax.yaxis.set_tick_params(color="w")
        cbps.outline.set_edgecolor("w")
        plt.setp(plt.getp(cbps.ax.axes, 'yticklabels'), color="w")
        ###
        fig1.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig1.savefig("./Figures/" + tit + "_poisson_{:.0f}.png".format(depth_interpolation.max()), dpi=600)
            fig1.savefig("./Figures/" + tit + "_poisson_{:.0f}.pdf".format(depth_interpolation.max()))
        
        
        ####################################################################################################
        ### PLOT VS 
        dminv = 6
        plot_subsurface(ax_vs, self.vs_interpolation, self.depth_interpolation_all, self.depth_interpolation.size, 
                              vs_min, vs_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminv )
        if do_truth:
            ax_vs.plot(vs_true_interpolation, self.depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10, label="True") #"11sta")#
        if do_ML:
            ax_vs.plot(profile_vs, depth_best, color=self.max_coldens, linewidth=2, zorder=1, label="Best")
        if do_MAP:
            if Nmap>1:
                for i in range(len(self.vs_map)):
                    # mapcol = colorskde[i]
                    ax_vs.plot(self.vs_map[i], self.depth_interpolation, color=cmap(i/Nmap), linewidth=2, ls="-.", zorder=100, label="MAP")
            else:
                ax_vs.plot(self.vs_map[0], self.depth_interpolation, color=self.map_coldens, linewidth=2, ls="-.", zorder=100, label="MAP")
        ###
        ax_vs.set_ylim(0, depth_max)
        ax_vs.set_xlim(vs_min,vs_max)
        ax_vs.set_xlabel('Shear velocity\n'+r'$v_s$ / [$km/s$]')
        ax_vs.set_ylabel(r'Depth / [$km$]')
        ax_vs.legend(loc=3, framealpha=0.3, facecolor="k", edgecolor='None', labelcolor="w")

        
        ####################################################################################################
        ### PLOT VP 
        hvp, valminvp, valmaxvp = plot_subsurface(ax_vp, self.vp_interpolation, self.depth_interpolation_all, self.depth_interpolation.size, 
                                                            vp_min, vp_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminv, aspect_ratio=aspect_ratio )

        if do_truth:
            ax_vp.plot(vp_true_interpolation, self.depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10)
        if do_ML:
            ax_vp.plot(profile_vp, depth_best, color=self.max_coldens, linewidth=2, zorder=1)
        if do_MAP:
            if Nmap>1:
                for i in range(len(self.vs_map)):
                    # mapcol = colorskde[i]
                    ax_vp.plot(self.vp_map[i], self.depth_interpolation, color=cmap(i/Nmap), linewidth=2, ls="-.", zorder=100, label="MAP")
            else:
                ax_vp.plot(self.vp_map[0], self.depth_interpolation, color=self.map_coldens, linewidth=2, ls="-.", zorder=100, label="MAP")
        
        ### COLORBAR VP 
        cbaxes = inset_axes(ax_vp, width="3%", height="30%", loc=3) 
        # cbvp = fig.colorbar(hvp, cax=cbaxes, orientation='vertical', ticks=[0, vmaxp] )
        cbvp = fig.colorbar(hvp, cax=cbaxes, orientation='vertical', ticks=[valminvp, valmaxvp] )
        cbvp.ax.set_yticklabels(['0', '{:d}'.format(dminv)])
        cbvp.set_label("Model/Mean Density", color="w")
        # cbvp.set_label("Probability", color="w")
        cbvp.ax.yaxis.set_tick_params(color="w")
        cbvp.outline.set_edgecolor("w")
        plt.setp(plt.getp(cbvp.ax.axes, 'yticklabels'), color="w")

        ###
        ax_vp.set_xlim(vp_min,vp_max)
        ax_vp.invert_yaxis()
        ax_vp.set_xlabel('Compressional velocity\n'+r'$v_p$ / [$km/s$]')
        # ax_vp.yaxis.set_ticks(labels=[])
        ax_vp.yaxis.set_tick_params(labelleft=False)


        ####################################################################################################
        ### HISTOGRAMS FOR SOURCE
        var = [t0s, source_depths]
        var_true = [t0_true, source_depth_true]
        if do_MAP: 
            var_map=[self.ts_map, self.hs_map]
        titvar = [r'Origin time / [$s$]', r'Source depth / [$km$]']
        wrs = [wr_ts, wr_hs]
        for ia, ax in enumerate(axs_oth[:2]) :
            ### True value 
            ax.axvline(var_true[ia], linewidth=2, color=self.true_col, zorder=1)
            ### Best value 
            ax.axvline(var[ia][idx_best], linewidth=1.8, ls="--", color=self.max_col, zorder=10)
            if do_MAP and wrs[ia] != []:
                if Nmap>1:
                    for i in range(len(self.vs_map)):
                        # mapcol = colorskde[i]
                        ax.axvline(var_map[ia][i], linewidth=1.8, ls="-.", color=cmap(i/Nmap), zorder=10)
                else:
                    ax.axvline(var_map[ia][0], linewidth=1.8, ls="-.", color=self.map_col, zorder=10)
            ### Calculate histogram bounded by prior limits.
            if not isinstance(wrs[ia], list): 
                phmin, phmax = self.prior_inv[wrs[ia]][0],self.prior_inv[wrs[ia]][1]
                ax.hist(var[ia], bins=bins_hist, color=self.hist_col, range=[phmin, phmax], 
                                alpha=self.hist_alpha, zorder=0, density=True, edgecolor='w', linewidth=0.5)
                ax.set_xlim(phmin, phmax)
            ###
            ax.set_xlabel(titvar[ia])
            ax.tick_params(axis='both', which='both', labelleft=False, left=False)
            ax.spines[['right', 'top', 'left']].set_visible(False)
        

        
        ### Find limits 
        qhist = np.percentile(self.flat_samples[:, wr_Ls[1]], [1, 99])
        qmin_lon = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
        qmax_lon = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
        ###
        qhist = np.percentile(self.flat_samples[:, wr_Ls[0]], [1, 99])
        qmin_lat = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
        qmax_lat = qhist[1] + 0.1*abs(qhist[1]-qhist[0])

        lats_all = self.DATA.sta_lats + [qmax_lat] + [qmin_lat] 
        lons_all = self.DATA.sta_lons + [qmax_lon] + [qmin_lon] 

        lat_bins = np.linspace(min(lats_all), max(lats_all), bins_hist)
        lon_bins = np.linspace(min(lons_all), max(lons_all), bins_hist)
        
        ### Get Posterior data and make 2D hist 
        hlat, hlon = self.flat_samples[:,wr_Ls[0]], self.flat_samples[:,wr_Ls[1]]
        axs_oth[2].hist2d(hlon, hlat, [lon_bins, lat_bins], cmap=self.loc_cmap, cmin=1 )#"PuRd"
               
        axs_oth[2].plot(self.DATA.ev_lon, self.DATA.ev_lat, markerfacecolor="#0081ff", #tcol, 
                                markeredgecolor="#0081ff", marker="*", markersize=7, markeredgewidth=1)  #w
        axs_oth[2].plot(source_lons[idx_best], source_lats[idx_best], fillstyle="none", 
                                markeredgecolor="w", marker="o", markersize=7, markeredgewidth=1)
        axs_oth[2].plot(source_lons[idx_best], source_lats[idx_best], fillstyle="none", 
                                markeredgecolor=self.max_coldens, marker="o", markersize=6, markeredgewidth=1)
        
        # Plot the stations
        for i , (slat, slon) in enumerate(zip(self.DATA.sta_lats, self.DATA.sta_lons)):
            axs_oth[2].plot(slon, slat, 'k', marker="^", markersize=5)
            axs_oth[2].text(slon, slat, '  {:d}'.format(i), ha='left', va='top')

        axs_oth[2].set_xlabel(r"Source Location / [$^{\circ}$Lat/Lon]")
        axs_oth[2].set_xlim(   (lon_bins.min()+lon_bins.max())/2 - 1.5*(lon_bins.max()-lon_bins.min())/2, 
                               (lon_bins.min()+lon_bins.max())/2 + 1.5*(lon_bins.max()-lon_bins.min())/2 )
        axs_oth[2].set_ylim(   (lat_bins.min()+lat_bins.max())/2 - 1.5*(lat_bins.max()-lat_bins.min())/2, 
                               (lat_bins.min()+lat_bins.max())/2 + 1.5*(lat_bins.max()-lat_bins.min())/2 )
        axs_oth[2].tick_params(axis="y",direction="in", pad=2,labelsize = fontsize*0.8)#, colors="w")pad=-17, 
        axs_oth[2].tick_params(axis="x",direction="in", labelsize = fontsize*0.8)
        axs_oth[2].set_facecolor(self.bgcol)
        axs_oth[2].patch.set_alpha(0.5)
        axs_oth[2].grid(lw=0.5, color="w", ls="--")
        axs_oth[2].set_axisbelow(True)
        # axs_oth[2].tick_params(axis="x",direction="in", pad=-17)


        ####################################################################################################
        ### PLOT ENSEMBLE OF GROUP VELOCITIES CURVES
        ax_disp.errorbar([], [], yerr=[], color=self.true_col, zorder=1, label='Picked vg', ls='', marker="s", markersize=2, elinewidth=1)
        for ii in range(Nwaves):
            if self.DATA.data_vector[ii]["RW_arrival"] is not None:
                dkm = self.DATA.dist_km[ii]
                dv = self.DATA.std_vector[ii]["RW_error"]*(dkm/(self.DATA.data_vector[ii]["RW_arrival"]-self.DATA.tprops_air[ii])**2)
                # ax_disp.plot(self.DATA.periods_RWs[ii], self.DATA.picked_vgs[ii], color=self.true_col, zorder=1, ls='--')
                ax_disp.errorbar(self.DATA.periods_RWs[ii], self.DATA.picked_vgs[ii], yerr=dv, 
                                        color=self.true_col, zorder=1, ls='', marker="s", markersize=2, elinewidth=1)
        
        for ig in range(Nmod):
            ax_disp.plot(periods, chosen_vgs[ig,:], c="k", alpha=0.05, lw=0.8)

        ### Remove true and Median for now (too busy) 
        # ax_disp.plot(self.DATA.periods, vg0_true[0], color=self.true_col, zorder=1, label='True', lw=2)
        # ax_disp.plot(periods, np.median(chosen_vgs, axis=0), color=self.max_coldens, alpha=1, zorder=9, label='Median')

        # dplus, dminus, frstd = [], [] , []
        # for ii in range(Nwaves):
        #     if self.DATA.data_vector[ii]["RW_arrival"] is not None:
        #         dkm = self.DATA.dist_km[ii]

        #         ### Uncertainty propagation 
        #         dv = self.DATA.std_vector[ii]["RW_error"]* (dkm/(self.DATA.data_vector[ii]["RW_arrival"]-self.DATA.tprops_air[ii])**2)
        #         dplus += list(dkm/self.DATA.data_vector[ii]["RW_arrival"]+dv)
        #         dminus += list(dkm/self.DATA.data_vector[ii]["RW_arrival"]-dv)
        #         frstd += list(self.DATA.periods_RWs[ii])
        # dplus = [x for _, x in sorted(zip(frstd, dplus ))]
        # dminus = [x for _, x in sorted(zip(frstd, dminus ))]
        # frstd = np.array(sorted(frstd))
        # # ax_disp.plot(frstd, dplus, c="darkturquoise", label=r"Data $\sigma$", lw=1.5, ls=":")
        # ax_disp.plot(frstd, dminus, c="darkturquoise", lw=1.5, ls=":")

        ### Plot best model 
        ax_disp.plot(periods, chosen_vgs[-1,:], color=self.max_col, zorder=10, label="Best", lw=2)
        ### Plot MAP
        if do_MAP:
            ax_disp.plot(periods, vg_map[0], color=self.map_coldens, zorder=10, ls="--", label="MAP", lw=2)
        ### 63% of models
        ax_disp.fill_between(periods, np.quantile(chosen_vgs, q=0.16, axis=0), np.quantile(chosen_vgs, q=0.84, axis=0), 
                                        color='black', alpha=0.2, zorder=8, label=r"Posterior 1-$\sigma$")
        

        ### Mean doesn't work if different data size 
        ax_disp.set_xlabel(r'Period / [$s$]')
        ax_disp.set_ylabel(r'Group Velocity / [$km/s$]')
        ax_disp.set_ylim(0.5,6.)
        ax_disp.yaxis.tick_right()
        ax_disp.yaxis.set_label_position("right")
        ax_disp.legend(framealpha=0.5, edgecolor='none', loc=2, ncol=2)
        ax_disp.grid(ls=':')
        ax_disp.set_xscale("log")
        ax_disp.set_xlim(np.min(periods), np.max(periods))


        ####################################################################################################
        ### PLOT WAVEFORM INFORMATION  
        for ii, dist_true in enumerate(self.DATA.dist_km):

            ### HISTOGRAM ON ORIGIN TIME 
            ax_ti = ax_waves[ii].twinx()
            ht, _, _ = ax_ti.hist(t0s, bins=np.linspace(-30, 30, bins_wave), color=self.hist_col, 
                                    alpha=self.hist_alpha, zorder=-100, edgecolor='w', lw=0.5, density=False, label="Posterior")
            ax_ti.spines[['right', 'top', 'left','bottom']].set_visible(False)
            ax_ti.set_yticks([])  
            ax_ti.set_ylim(-ht.max(),ht.max())
            if ii<Nwaves-1:  
                ax_ti.set_xticks([])   
            ### Ensures t0 histogram has same height as S histogram
            # ax_waves[ii].bar(x=t0bins[:-1], height=t0hist/t0hist.max()*shist.max(), align="edge",
            #                     width = np.diff(t0bins),
            #                     color=self.hist_col, alpha=self.hist_alpha, zorder=-1, edgecolor='w', linewidth=0.5    )
            ax_ti.axvline(t0_true, linewidth=1.5, ls="-.", color=self.true_col, zorder=2)
            ax_ti.axvline(t0s[-1], linewidth=1, ls="-", color=self.max_col, zorder=10)
            
            tair = self.DATA.tprops_air[ii]
                
            ### PLOT WAVEFORM-RELATED INFORMATION
            sphistmax = 1
            if self.DATA.data_vector[ii]["S_arrival"] != None:
                ### OPTION 1: Show arrival expected from best model 
                # true_S = compute_travel_time_SP(dist_true, source_depth_true, 
                #                                     self.DATA.true_velocity_model, phase='s')
                # ### We account for origin time and propagation time in air: 
                # true_arr_S = t0_true + true_S + tair
                # ### Chosen_S was already converted to an arrival time. 
                # ax_waves[ii].axvline(true_arr_S, linewidth=1.5, color=self.true_col, zorder=2)#, label="True S")

                ### OPTION 2: Show picked arrival 
                ax_waves[ii].axvline(self.DATA.data_vector[ii]["S_arrival"], linewidth=1.5, ls="--", color=self.true_col, zorder=2)

                ### Histogram 
                shist, _, _ = ax_waves[ii].hist(chosen_tS[ii,:], bins=np.linspace(self.DATA.data_vector[ii]["S_arrival"]/2, 1.5*self.DATA.data_vector[ii]["S_arrival"], bins_wave), 
                                                color=self.hist_col, alpha=self.hist_alpha, 
                                                zorder=-1, edgecolor='w', lw=0.5, density=False)
                ax_waves[ii].axvline(chosen_tS[ii,-1], linewidth=1, color=self.max_col, zorder=10)
                sphistmax = max(shist.max(), sphistmax)

            if self.DATA.data_vector[ii]["P_arrival"] != None:
                ### OPTION 1: Show arrival expected from best model 
                # true_P = compute_travel_time_SP(dist_true, source_depth_true, 
                #                                     self.DATA.true_velocity_model, phase='p')
                # ### We account for origin time and propagation time in air: 
                # true_arr_P = t0_true + true_P + tair
                # ### Chosen_S was already converted to an arrival time. 
                # ax_waves[ii].axvline(true_arr_P, linewidth=1.5, color=self.true_col, zorder=2)#, label="True P")

                ### OPTION 2: Show picked arrival 
                ax_waves[ii].axvline(self.DATA.data_vector[ii]["P_arrival"], linewidth=1.5, color=self.true_col, zorder=2)

                ### Histogram 
                phist, _, _ = ax_waves[ii].hist(chosen_tP[ii,:], bins=np.linspace(self.DATA.data_vector[ii]["P_arrival"]/2, 1.5*self.DATA.data_vector[ii]["P_arrival"], bins_wave), 
                                                color=self.hist_col, alpha=self.hist_alpha, 
                                                zorder=-1, edgecolor='w', lw=0.5, density=False)
                ax_waves[ii].axvline(chosen_tP[ii,-1], linewidth=1, color=self.max_col, zorder=10)
                sphistmax = max(phist.max(), sphistmax)


            ### RETRIEVE WAVEFORM FOR INDEX
            waveform = self.DATA.traces[ii]
            ### Waveform signals 
            if self.DATA.key_traces == 1:
                waveform.taper(type ="hann", max_percentage = 0.1)
                # waveform.filter("bandpass", freqmin = 8e-3, freqmax = 0.99*5e-1)
                waveform.filter("bandpass", freqmin = 5e-2, freqmax = 0.99*5e-1)
                time, amp = waveform.times(reftime=self.DATA.t_event), waveform.data
            else:
                time, amp = waveform.get_xdata(), waveform.get_ydata()
            ax_waves[ii].plot(time, sphistmax*amp/amp.max(), color='black', zorder=1, lw=0.5)

            ### Measurement error on S pick. 
            if self.DATA.data_vector[ii]["S_arrival"] != None:
                ### Measured arrival time already accounts for an origin time and air propagation 
                gauss = sphistmax*np.exp(-((self.DATA.data_vector[ii]["S_arrival"]-time)/self.DATA.std_vector[ii]["S_error"])**2)
                ax_waves[ii].fill_between(time, gauss, color=self.true_col, zorder=-1, alpha=0.4, linewidth=0.5)
            ### Measurement error on P pick. 
            if self.DATA.data_vector[ii]["P_arrival"] != None:
                ### Measured arrival time already accounts for an origin time and air propagation 
                gauss = sphistmax*np.exp(-((self.DATA.data_vector[ii]["P_arrival"]-time)/self.DATA.std_vector[ii]["P_error"])**2)
                ax_waves[ii].fill_between(time, gauss, color=self.true_col, zorder=-1, alpha=0.4, linewidth=0.5)
            ax_waves[ii].set_ylim(-1.*sphistmax, 1.*sphistmax)


        ####################################################################################################
        ### FINAL EDITS ON FIGURE  
        ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col, ls="-.", zorder=2, label="Origin")
        ax_waves[0].hist([], bins=1, color=self.hist_col, alpha=0.6, zorder=-100, edgecolor='w', lw=0.5, density=False, label="Posterior")
        if do_S : 
            ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col,ls="--", zorder=2, label="Picked S")
        ax_waves[0].axvline(-1e9, linewidth=1, color=self.max_col, zorder=10, label="Best")
        if do_P:
            ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col, zorder=2, label="Picked P")
                
        for ii in range(Nwaves):
            ax_waves[ii].yaxis.tick_right()
            ax_waves[ii].yaxis.set_label_position("right")
            ax_waves[ii].set_yticks([])    
            ax_waves[ii].tick_params(axis='both', which='both', labelright=False, left=False)
            ax_waves[ii].spines[['right', 'top', 'left']].set_visible(False)
            ax_waves[ii].set_xlim(-100, 1000)
            ax_waves[ii].text(-100, ax_waves[ii].get_ylim()[1], '{:d}'.format(ii), ha='left', va='top', 
                                bbox=dict(facecolor='white', pad=0, linewidth=0))
            if ii<Nwaves-1:
                ax_waves[ii].set_xticks([])
                ax_waves[ii].spines[['bottom']].set_visible(False)
        ax_waves[-1].set_xlabel(r'Time / [$s$]')
        ax_waves[Nwaves//2].set_ylabel(r'Waveforms, origin, S & P arrivals')
        ax_waves[0].legend(framealpha=1, edgecolor='None', loc='lower center', bbox_to_anchor=(0.5, 1.3),ncol=3)
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig.savefig("./Figures/" + tit + "_model_{:.0f}.png".format(depth_interpolation.max()), dpi=600)
            fig.savefig("./Figures/" + tit + "_model_{:.0f}.pdf".format(depth_interpolation.max()))


    ##########################################################################################################  
    ### VISUALIZE PRIORS
    ##########################################################################################################  
    def visualize_priors(self, figsize=(10,5), Nmod= 1000, Nvg=200, bins_hist=20, bins_wave=40, hspace=2.5, wspace=0.1, 
                            depth_interpolation=np.linspace(0., 150., 201), do_save=False, preserve_aspect=False, 
                            color_code=1
    ):

        ####################################################################################################
        ### Get Model information 
        wr_ts = self.wr_order[0]
        wr_Ls = self.wr_order[1]
        wr_hs = self.wr_order[2]
        wr_vs = self.wr_order[3]
        wr_ps = self.wr_order[4]
        wr_hl = self.wr_order[5]
        Nwaves =  len(self.DATA.traces)

        ####################################################################################################
        ### Load true solution
        periods = self.DATA.target_periods
        t0_true, _, _, source_depth_true = self.DATA.truth_location


        ####################################################################################################
        ### Pick random models among prior bounds: 
        l_bounds = [s[0] for s in self.prior_lims]  
        u_bounds = [s[1] for s in self.prior_lims] 
        prior_models = np.zeros((len(l_bounds), Nmod ))
        for i, (a, b) in enumerate(zip(l_bounds, u_bounds)): 
            prior_models[i,:] = np.random.uniform(low=a, high=b, size=Nmod)

        ### DOWNSELECT ACCORDING TO LOG_PRIOR (REQUIRES QUITE A LOT OF MODELS INITIALY !) 
        is_ok = np.array([log_prior(x, self.prior_inv, self.wr_order) for x in prior_models.T])
        true_prior_models = prior_models[:,np.where(is_ok==0)[0]].T 
        Nmod = true_prior_models.shape[0] 

        ####################################################################################################
        ### INTERPOLATE PRIORS THIS TIME 
        N_intp = depth_interpolation.size

        ### Smoother transitions at interfaces
        vs_interpolation = np.zeros((Nmod, N_intp))
        vp_interpolation = np.zeros((Nmod, N_intp))
        ps_interpolation = np.zeros((Nmod, N_intp))

        print("Interpolating prior models...")
        for i in tqdm(range(Nmod),ncols=80, leave=True, position=0, file=sys.stdout):
            
            vs = true_prior_models[i,wr_vs]
            poisson = true_prior_models[i,wr_ps]
            h_layer = true_prior_models[i,wr_hl]

            velocity_model = create_model(vs, poisson, h_layer)
            vs_interpolation[i,:] = build_model_from_thicknesses(velocity_model, depth_interpolation, 2)
            vp_interpolation[i,:] = build_model_from_thicknesses(velocity_model, depth_interpolation, 1)
            ps_interpolation[i,:] = build_model_from_thicknesses(velocity_model, depth_interpolation, 3)
            
        vs_true_interpolation = build_model_from_thicknesses(self.DATA.true_velocity_model, depth_interpolation, 2)
        vp_true_interpolation = build_model_from_thicknesses(self.DATA.true_velocity_model, depth_interpolation, 1)
        ps_true_interpolation = build_model_from_thicknesses(self.DATA.true_velocity_model, depth_interpolation, 3)
            
        vs_interpolation=vs_interpolation.flatten()
        vp_interpolation=vp_interpolation.flatten()
        ps_interpolation=ps_interpolation.flatten()
        depth_interpolation_all = np.tile(depth_interpolation, Nmod)


        ####################################################################################################
        ### Get source information 
        t0s = true_prior_models[:,wr_ts]
        source_lats, source_lons = true_prior_models[:,wr_Ls[0]], true_prior_models[:,wr_Ls[1]]
        source_depths = true_prior_models[:,wr_hs]


        ####################################################################################################
        ### CALCULATE WAVE PARAMETERS FROM PRIORS 
        ### Pick Nmod random models to display group velocity curves.
        chosen_id = np.random.choice(Nmod, size=Nvg, replace=False)
        chosen_vgs = np.zeros((Nvg, periods.size))
        if np.any( [self.DATA.data_vector[i]["S_arrival"] for i in range(Nwaves)] is not None):
            chosen_tS = np.zeros((Nwaves,Nmod+1))
            do_S = True 
        if np.any( [self.DATA.data_vector[i]["P_arrival"] for i in range(Nwaves)] is not None):
            chosen_tP = np.zeros((Nwaves,Nmod+1))
            do_P = True 
        print("Calculating group velocity and arrival times...")
        for ic, idx in enumerate(tqdm(chosen_id, ncols=80, leave=True, position=0, file=sys.stdout)):
            chosen_vs = true_prior_models[idx,wr_vs]
            chosen_poisson = true_prior_models[idx,wr_ps]
            chosen_h_layer = true_prior_models[idx,wr_hl]

            ### Calculate vg 
            chosen_velocity_model = create_model(chosen_vs, chosen_poisson, chosen_h_layer)
            _, chosen_vg = compute_vg_n_layers(periods, chosen_velocity_model, max_mode = 1)
            if chosen_vg is not None:
                u = chosen_vg[0].size
                chosen_vgs[ic,:u] = chosen_vg[0]

            ### Calculate S arrival time 
            chosen_lat, chosen_lon = source_lats[idx], source_lons[idx]
            chosen_source_depth = source_depths[idx]
            chosen_t0 = t0s[idx]
            ### Loop on stations 
            if do_S or do_P:
                for il, (sta_lat, sta_lon) in enumerate(zip(self.DATA.sta_lats, self.DATA.sta_lons)):
                    if self.DATA.data_vector[il]["S_arrival"] is not None or self.DATA.data_vector[il]["P_arrival"] is not None:    
                        sta_dist = gps2dist_azimuth(sta_lat, sta_lon, chosen_lat, chosen_lon)[0]/1e3

                        if self.DATA.data_vector[il]["S_arrival"] is not None:
                            tS = compute_travel_time_SP(  sta_dist, chosen_source_depth, chosen_velocity_model, phase='s' )
                            ### We convert the travel time to an arrival time: 
                            chosen_tS[il,ic] = tS + chosen_t0 + self.DATA.tprops_air[il] 

                        if self.DATA.data_vector[il]["P_arrival"] is not None:
                            tP = compute_travel_time_SP(  sta_dist, chosen_source_depth, chosen_velocity_model, phase='p' ) 
                            ### We convert the travel time to an arrival time: 
                            chosen_tP[il,ic] = tP + chosen_t0 + self.DATA.tprops_air[il] 


        ####################################################################################################
        ### MAKE PLOT 
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=0.07, bottom=0.12, right=0.95, top=0.95)

        ### COLOR PROFILE FOR FIGURE 
        #############################################################################################################
        if not hasattr(self, "true_col"):
            self.color_palette(color_code=color_code)

        ### changes the fontsize of matplotlib, not just a single figure
        fontsize=8/10*figsize[0]
        rcParams.update({'font.size': fontsize})
        ###
        ### Axis limits for vp and vs profiles 
        vs_min, vs_max = min([self.prior_inv[i][0] for i in wr_vs]), max([self.prior_inv[i][1] for i in wr_vs])
        ps_min, ps_max = min([self.prior_inv[i][0] for i in wr_ps]), max([self.prior_inv[i][1] for i in wr_ps])
        vp_min, vp_max = vp_from_vs(vs_min, ps_min), vp_from_vs(vs_max, ps_max)
        ###
        if preserve_aspect:
            aspect_ratio = (vp_max-vp_min)/(vs_max-vs_min)
            grid = fig.add_gridspec(6*Nwaves+2*Nwaves, 5, width_ratios=[1,aspect_ratio,1,1,1])
        else:
            aspect_ratio = None
            grid = fig.add_gridspec(6*Nwaves+2, 5)
        ### Axes for seismic model results 
        ax_vs = fig.add_subplot(grid[:,0])
        ax_vp = fig.add_subplot(grid[:,1], sharey=ax_vs)
        ax_t0 = fig.add_subplot(grid[:2*Nwaves, 2])
        ax_dist = fig.add_subplot(grid[3*Nwaves:5*Nwaves, 2])
        ax_depth = fig.add_subplot(grid[6*Nwaves:, 2])
        ax_disp = fig.add_subplot(grid[:3*Nwaves,-2:])
        ax_waves = [] 
        for iwave in range(Nwaves):
            ax_waves.append(fig.add_subplot(grid[5*Nwaves+iwave*3 :5*Nwaves+3*(1+iwave),3:]))
        axs_oth = [ax_t0,ax_dist,ax_depth, ax_disp] + ax_waves
        depth_max = 0.999*depth_interpolation.max()  


        ####################################################################################################
        ### PLOT AND SAVE POISSON RATIO IN SEPARATE PLOT 
        fig1 = plt.figure(figsize=(figsize[0]/5,figsize[1]))
        fig1.subplots_adjust(hspace=hspace, wspace=wspace, left=0.03, bottom=0.12, right=0.95, top=0.95)
        ax_ps = fig1.add_subplot(111)
        dminps = 2

        hvac, valminps, valmaxps = plot_subsurface(ax_ps, ps_interpolation, depth_interpolation_all, depth_interpolation.size, 
                                                            ps_min, ps_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminps)
        ax_ps.plot(ps_true_interpolation, depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10, label="True")
        
        ### DECOR 
        ax_vs.set_ylim(0, depth_max)
        ax_vs.set_xlim(ps_min,ps_max)
        ax_ps.set_xlabel("Poisson's ratio")
        ax_ps.set_ylabel(r'Depth / [$km$]')
        ax_ps.legend(loc=4, framealpha=0, edgecolor='None', labelcolor="w")
        ax_ps.invert_yaxis()
        
        ### COLORBAR PS 
        cbaxes1 = inset_axes(ax_ps, width="8%", height="30%", loc=3) 
        cbps = fig.colorbar(hvac, cax=cbaxes1, orientation='vertical', ticks=[valminps, valmaxps] )
        cbps.ax.set_yticklabels(['0', '{:d}'.format(dminps)])
        cbps.set_label("Model/Mean Density", color="w")
        cbps.ax.yaxis.set_tick_params(color="w")
        cbps.outline.set_edgecolor("w")
        plt.setp(plt.getp(cbps.ax.axes, 'yticklabels'), color="w")
        ###
        fig1.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig1.savefig("./Figures/" + tit + "_poisson¨_priormodel.png", dpi=600)
            fig1.savefig("./Figures/" + tit + "_poisson¨_priormodel.pdf")
        
        
        ####################################################################################################
        ### PLOT VS 
        dminv = 6
        plot_subsurface(ax_vs, vs_interpolation, depth_interpolation_all, depth_interpolation.size, 
                              vs_min, vs_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminv )

        ax_vs.plot(vs_true_interpolation, depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10, label="True")
        ###
        ax_vs.set_ylim(0, depth_max)
        ax_vs.set_xlim(vs_min,vs_max)
        ax_vs.set_xlabel('Shear velocity\n'+r'$v_s$ / [$km/s$]')
        ax_vs.set_ylabel(r'Depth / [$km$]')
        ax_vs.legend(loc=3, framealpha=0, edgecolor='None', labelcolor="w")

        
        ####################################################################################################
        ### PLOT VP 
        hvp, valminvp, valmaxvp = plot_subsurface(ax_vp, vp_interpolation, depth_interpolation_all, depth_interpolation.size, 
                                                            vp_min, vp_max, depth_max, self.dens_cmap, self.bgcol, dmin= dminv, aspect_ratio=aspect_ratio )

        ax_vp.plot(vp_true_interpolation, depth_interpolation, color=self.true_coldens, linewidth=3, zorder=10)
        
        ### COLORBAR VP 
        cbaxes = inset_axes(ax_vp, width="3%", height="30%", loc=3) 
        # cbvp = fig.colorbar(hvp, cax=cbaxes, orientation='vertical', ticks=[0, vmaxp] )
        cbvp = fig.colorbar(hvp, cax=cbaxes, orientation='vertical', ticks=[valminvp, valmaxvp] )
        cbvp.ax.set_yticklabels(['0', '{:d}'.format(dminv)])
        cbvp.set_label("Model/Mean Density", color="w")
        # cbvp.set_label("Probability", color="w")
        cbvp.ax.yaxis.set_tick_params(color="w")
        cbvp.outline.set_edgecolor("w")
        plt.setp(plt.getp(cbvp.ax.axes, 'yticklabels'), color="w")

        ###
        ax_vp.set_xlim(vp_min,vp_max)
        ax_vp.invert_yaxis()
        ax_vp.set_xlabel('Compressional velocity\n'+r'$v_p$ / [$km/s$]')
        # ax_vp.yaxis.set_ticks(labels=[])
        ax_vp.yaxis.set_tick_params(labelleft=False)

        ### New: Source time, source depth, location map 
        var = [t0s, source_depths]
        var_true = [t0_true, source_depth_true]
        titvar = [r'Origin time / [$s$]', r'Source depth / [$km$]']
        wrs = [wr_ts, wr_hs]
        for ia, ax in enumerate(axs_oth[:2]) :
            ### True value 
            ax.axvline(var_true[ia], linewidth=2, color=self.true_col, zorder=1)
            ### Calculate histogram bounded by prior limits.
            phmin, phmax = self.prior_inv[wrs[ia]][0],self.prior_inv[wrs[ia]][1]
            ax.hist(var[ia], bins=bins_hist, color=self.hist_col, range=[phmin, phmax], 
                            alpha=self.hist_alpha, zorder=0, density=True, edgecolor='w', linewidth=0.5)
            ###
            ax.set_xlabel(titvar[ia])
            ax.set_xlim(phmin, phmax)
            ax.tick_params(axis='both', which='both', labelleft=False, left=False)
            ax.spines[['right', 'top', 'left']].set_visible(False)
        
        ### Find limits 
        lat_bins = np.linspace(self.prior_inv[wr_Ls[0]][0], self.prior_inv[wr_Ls[0]][1], bins_hist)
        lon_bins = np.linspace(self.prior_inv[wr_Ls[1]][0], self.prior_inv[wr_Ls[1]][1], bins_hist)

        ### Get Posterior data and make 2D hist 
        hlat, hlon = source_lats, source_lons
        axs_oth[2].hist2d(hlon, hlat, [lon_bins, lat_bins], cmap=self.loc_cmap, cmin=1 )#"PuRd"
               
        axs_oth[2].plot(self.DATA.ev_lon, self.DATA.ev_lat, markerfacecolor="#0081ff", #tcol, 
                                markeredgecolor="#0081ff", marker="*", markersize=7, markeredgewidth=1)
        
        # Plot the stations
        for i , (slat, slon) in enumerate(zip(self.DATA.sta_lats, self.DATA.sta_lons)):
            axs_oth[2].plot(slon, slat, 'k', marker="^", markersize=5)
            axs_oth[2].text(slon, slat, '  {:d}'.format(i), ha='left', va='top')

        axs_oth[2].set_xlabel(r"Source Location / [$^{\circ}$Lat/Lon]")
        axs_oth[2].set_xlim(  lon_bins.min(), lon_bins.max())
        axs_oth[2].set_ylim(  lat_bins.min(), lat_bins.max())
        axs_oth[2].tick_params(axis="y",direction="in", pad=2,labelsize = fontsize*0.8)#, colors="w")pad=-17, 
        axs_oth[2].tick_params(axis="x",direction="in", labelsize = fontsize*0.8)
        axs_oth[2].set_facecolor(self.bgcol)
        axs_oth[2].patch.set_alpha(0.5)
        axs_oth[2].grid(lw=0.5, color="w", ls="--")
        axs_oth[2].set_axisbelow(True)

        ####################################################################################################
        ### PLOT ENSEMBLE OF GROUP VELOCITIES CURVES

        ax_disp.errorbar([], [], yerr=[], color=self.true_col, zorder=1, label='Picked vg', ls='', marker="s", markersize=2, elinewidth=1)
        for ii in range(Nwaves):
            if self.DATA.data_vector[ii]["RW_arrival"] is not None:
                dkm = self.DATA.dist_km[ii]
                dv = self.DATA.std_vector[ii]["RW_error"]*(dkm/(self.DATA.data_vector[ii]["RW_arrival"]-self.DATA.tprops_air[ii])**2)
                # ax_disp.plot(self.DATA.periods_RWs[ii], self.DATA.picked_vgs[ii], color=self.true_col, zorder=1, ls='--')
                ax_disp.errorbar(self.DATA.periods_RWs[ii], self.DATA.picked_vgs[ii], yerr=dv, 
                                        color=self.true_col, zorder=1, ls='', marker="s", markersize=2, elinewidth=1)
        
        for ig in range(Nvg):
            ax_disp.plot(periods, chosen_vgs[ig,:], c="k", alpha=0.05, lw=0.8)

        ### 63% of models
        ax_disp.fill_between(periods, np.quantile(chosen_vgs, q=0.16, axis=0), np.quantile(chosen_vgs, q=0.84, axis=0), 
                                        color='black', alpha=0.2, zorder=8, label=r"Posterior 1-$\sigma$")
        
        ### Mean doesn't work if different data size 
        ax_disp.set_xlabel(r'Period / [$s$]')
        ax_disp.set_ylabel(r'Group Velocity / [$km/s$]')
        ax_disp.set_ylim(0.5,6.)
        ax_disp.yaxis.tick_right()
        ax_disp.yaxis.set_label_position("right")
        ax_disp.legend(framealpha=0.5, edgecolor='none', loc=2, ncol=2)
        ax_disp.grid(ls=':')
        ax_disp.set_xscale("log")
        ax_disp.set_xlim(np.min(periods), np.max(periods))


        ####################################################################################################
        ### PLOT WAVEFORM INFORMATION  
        for ii, dist_true in enumerate(self.DATA.dist_km):
            ### HISTOGRAM ON ORIGIN TIME 
            ax_ti = ax_waves[ii].twinx()
            ht, _, _ = ax_ti.hist(t0s, bins=np.linspace(-30, 30, bins_wave), color=self.hist_col, 
                                    alpha=self.hist_alpha, zorder=-100, edgecolor='w', lw=0.5, density=False)
            ax_ti.spines[['right', 'top', 'left','bottom']].set_visible(False)
            ax_ti.set_yticks([])  
            ax_ti.set_ylim(-ht.max(),ht.max())
            if ii<Nwaves-1:  
                ax_ti.set_xticks([])   
            ax_ti.axvline(t0_true, linewidth=1.5, ls="-.", color=self.true_col, zorder=2)
            
            tair = self.DATA.tprops_air[ii]

            ### PLOT WAVEFORM-RELATED INFORMATION
            sphistmax=1
            if self.DATA.data_vector[ii]["S_arrival"] != None:
                ### Show picked arrival 
                ax_waves[ii].axvline(self.DATA.data_vector[ii]["S_arrival"], linewidth=1.5, ls="--", color=self.true_col, zorder=2)

                ### Histogram 
                tSs = chosen_tS[ii,:]
                tSs = tSs[tSs<1e9]
                shist, _, _ = ax_waves[ii].hist(chosen_tS[ii,:], bins=np.linspace(self.DATA.data_vector[ii]["S_arrival"]/2, 1.5*self.DATA.data_vector[ii]["S_arrival"], bins_wave), 
                                                color=self.hist_col, alpha=self.hist_alpha, 
                                                zorder=-1, edgecolor='w', lw=0.5, density=False)
                sphistmax = sphistmax if np.isnan(np.max(shist)) else max(sphistmax, np.max(shist))

            if self.DATA.data_vector[ii]["P_arrival"] != None:
                ### Show picked arrival 
                ax_waves[ii].axvline(self.DATA.data_vector[ii]["P_arrival"], linewidth=1.5, color=self.true_col, zorder=2)

                ### Histogram 
                tPs = chosen_tP[ii,:]
                tPs = tPs[tPs<1e9]
                phist, _, _ = ax_waves[ii].hist(tPs, bins=np.linspace(self.DATA.data_vector[ii]["P_arrival"]/2, 1.5*self.DATA.data_vector[ii]["P_arrival"], bins_wave), 
                                                color=self.hist_col, alpha=self.hist_alpha, 
                                                zorder=-1, edgecolor='w', lw=0.5, density=False)
                sphistmax = sphistmax if np.isnan(np.max(phist)) else max(sphistmax, np.max(phist))

            ### RETRIEVE WAVEFORM FOR INDEX
            waveform = self.DATA.traces[ii]
            ### Waveform signals 
            if self.DATA.key_traces == 1:
                waveform.taper(type ="hann", max_percentage = 0.1)
                waveform.filter("bandpass", freqmin = 8e-3, freqmax = 2)
                time, amp = waveform.times(reftime=self.DATA.t_event), waveform.data
            else:
                time, amp = waveform.get_xdata(), waveform.get_ydata()
            ax_waves[ii].plot(time, sphistmax*amp/amp.max(), color='black', zorder=1, lw=0.8)

            ### Measurement error on S pick. 
            if self.DATA.data_vector[ii]["S_arrival"] != None:
                ### Measured arrival time already accounts for an origin time and air propagation 
                gauss = sphistmax*np.exp(-((self.DATA.data_vector[ii]["S_arrival"]-time)/self.DATA.std_vector[ii]["S_error"])**2)
                ax_waves[ii].fill_between(time, gauss, color=self.true_col, zorder=-1, alpha=0.4, linewidth=0.5)
            ### Measurement error on P pick. 
            if self.DATA.data_vector[ii]["P_arrival"] != None:
                ### Measured arrival time already accounts for an origin time and air propagation 
                gauss = sphistmax*np.exp(-((self.DATA.data_vector[ii]["P_arrival"]-time)/self.DATA.std_vector[ii]["P_error"])**2)
                ax_waves[ii].fill_between(time, gauss, color=self.true_col, zorder=-1, alpha=0.4, linewidth=0.5)    
            ax_waves[ii].set_ylim(-1.*sphistmax, 1.*sphistmax)


        ####################################################################################################
        ### FINAL EDITS ON FIGURE  
        ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col, ls="-.", zorder=2, label="Origin")
        ax_waves[0].hist([], bins=1, color=self.hist_col, alpha=0.6, zorder=-100, edgecolor='w', lw=0.5, density=False, label="Posterior")
        if do_S : 
            ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col,ls="--", zorder=2, label="Picked S")
        ax_waves[0].axvline(-1e9, linewidth=1, color=self.max_col, zorder=10, label="Best")
        if do_P:
            ax_waves[0].axvline(-1e9, linewidth=1.5, color=self.true_col, zorder=2, label="Picked P")

        for ii in range(Nwaves):
            ax_waves[ii].yaxis.tick_right()
            ax_waves[ii].yaxis.set_label_position("right")
            ax_waves[ii].set_yticks([])    
            ax_waves[ii].tick_params(axis='both', which='both', labelright=False, left=False)
            ax_waves[ii].spines[['right', 'top', 'left']].set_visible(False)
            ax_waves[ii].set_xlim(-100,1000)
            ax_waves[ii].text(-100, ax_waves[ii].get_ylim()[1], '{:d}'.format(ii), ha='left', va='top', 
                                bbox=dict(facecolor='white', pad=0, linewidth=0))
            if ii<Nwaves-1:
                ax_waves[ii].set_xticks([])
                ax_waves[ii].spines[['bottom']].set_visible(False)
        ax_waves[-1].set_xlabel(r'Time / [$s$]')
        ax_waves[Nwaves//2].set_ylabel(r'Waveforms, origin, S & P arrivals')
        ax_waves[0].legend(framealpha=1, edgecolor='None', loc='lower center', bbox_to_anchor=(0.5, 1.3),ncol=3)
         
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig.savefig("./Figures/" + tit + "_priormodel_{:.0f}.png".format(depth_interpolation.max()), dpi=600)
            fig.savefig("./Figures/" + tit + "_priormodel_{:.0f}.pdf".format(depth_interpolation.max()))


    ##########################################################################################################  
    ### MAKE INDIVIDUAL HISTOGRAMS
    ##########################################################################################################  
    def parameter_histograms(self, figsize=(4,3), bins_hist=40, hspace=0.2, wspace=0.2, find_limits=True, do_plot_contours=False, 
                             do_save=False, do_MAP=True, do_truth=False):
        
        if find_limits: 
            limsc = [] 
        else: 
            limsc = self.prior_inv
        
        if not hasattr(self, "true_col"):
            self.color_palette(color_code=1)

        if do_MAP:
            if not hasattr(self, 'map_estimate'):
                self.get_MAP(NM=self.n_end) 


        ####################################################################################################
        ### Get Model information 
        wr_ts = self.wr_order[0]
        wr_Ls = self.wr_order[1]
        wr_hs = self.wr_order[2]
        wr_vs = self.wr_order[3]
        wr_ps = self.wr_order[4]
        wr_hl = self.wr_order[5]
        wrall_hl = self.wrall_order[5]

        Nparam =  self.flat_samples.shape[1]
        truth_location = self.DATA.truth_location

        ####################################################################################################
        ### Load true solution
        t0_true, lat_true, lon_true, source_depth_true = self.DATA.truth_location


        ####################################################################################################
        ### Find best solution
        idx_best = np.argmax(self.log_flat_samples)


        ####################################################################################################
        ### PREPARE PLOT 
        ### Test different cmaps
        hcol = "crimson"  #'tab:green'
        tcol = 'navy'  #'tab:blue'
        mcol = 'crimson' #'tab:green'
        mcol2 = "lightpink"
        tcol2 = "mediumslateblue"
        colkde = "#00d294"
        my_cmap = sns.color_palette("crest", as_cmap=True)
        #my_cmap.set_under('k',0)
        
        ### changes the fontsize of matplotlib, not just a single figure
        fontsize=80/30*figsize[0]
        rcParams.update({'font.size': fontsize})


        ####################################################################################################
        ### MAKE PLOT FOR EACH PARAMETER U 
        for u in range(Nparam):
            fig, ax = plt.subplots(figsize=figsize)

            qvalues = np.percentile(self.flat_samples[:, u], [16, 84])
            if find_limits:
                qhist = np.percentile(self.flat_samples[:, u], [1, 99])
                qmin = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
                qmax = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
                ### If the spread is large, just use the priors.
                if abs(qmax-qmin) >= abs(self.prior_inv[u][1] - self.prior_inv[u][0])/3:
                    limsc.append(self.prior_inv[u])
                else:
                    limsc.append([qmin, qmax])
                
            ##########################    
            ### Make the 1d histogram 
            # ax.text(s="u={:d}, v={:d}".format(u,v), x=0, y=0, ha="center")
            ha = ax.hist(self.flat_samples[:, u], bins_hist, range=(limsc[u][0],limsc[u][1]),
                                    color=hcol, alpha=0.4, zorder=0, histtype="bar",edgecolor='w',linewidth=0.5)
            if do_truth:
                ax.axvline(truth_inverted[u], color=tcol, linewidth = np.sqrt(Nparam/5), label="True")
            ax.axvline(self.flat_samples[idx_best,u], color=mcol, linewidth = np.sqrt(Nparam/5), ls=':', label="Best")
            if do_MAP:
                ax.axvline(self.map_estimate[u], color=colkde, linewidth = np.sqrt(Nparam/5), ls=':', label="MAP")
            # if u>3 and u<9:
            #     print(ha, limsc[u][0],limsc[u][1])


            ############################################################
            ### Axis decoration 
            ax.set_xlim(limsc[u][0], limsc[u][1])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.locator_params(axis="x", nbins=4)
            ax.set_xlabel(self.names[u])#.replace(' [', '\n['))

            # ax.yaxis.tick_right()
            # ax.yaxis.set_label_position("right")
            ax.set_ylabel('Count')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax.locator_params(axis="y", nbins=1)    
   
            ax.tick_params(axis='both', which='both', labelsize=0.8*fontsize)
            ax.legend(loc=0, framealpha=0.5, edgecolor="None")
            
            fig.tight_layout()
            if do_save:
                tit = self.method + self.run_name
                if not find_limits: 
                    tit += '_prior'
                fig.savefig("./Figures/" + tit + "_hist2d_{:d}.png".format(u), dpi=600)
                fig.savefig("./Figures/" + tit + "_hist2d_{:d}.pdf".format(u))
            # fig.clf()


        ####################################################################################################
        ### MAKE PLOT FOR INTERFACE DEPTH 
        if wr_hl.size>1:
            ####################################################################################################
            ### GET INFORMATION ON INTERFACE DEPTH 
            samples_depthlayer, prior_at_depth, mixed_priors, limsc_depth, map_model_depth = self.make_hist_depth(self.flat_samples, self.blobs, self.map_estimate)
            maxmax = np.sum([self.prior_inv[u][1] for u in wr_hl])  
            minmin = self.prior_min_sorted[5][0]      
            all_interfaces = np.concatenate([samples_depthlayer[:,w] for w in wr_hl])
            all_priors =  np.concatenate([prior_at_depth[:,w] for w in range(wr_hl.size)])  
            all_mixed_priors =  np.concatenate([mixed_priors[:,w] for w in range(wr_hl.size)])   
                    
                
            ##############################
            #############################################################################################
            fig2, (axc, axm) = plt.subplots(2,1,figsize=(6,3))   
            mincrust, maxcrust = 0, 175
            minmantle, maxmantle = 175,maxmax  
            hpc = axc.hist(all_priors, bins=bins_hist*2, range=[mincrust,maxcrust], histtype = "step", color='k', alpha=0.8, density=True)
            hppc = axc.hist(all_interfaces, bins=bins_hist*2, range=[mincrust,maxcrust], color=self.hist_col, alpha=0.4, zorder=0, 
                                        histtype="bar",edgecolor='w',linewidth=0.5, density=True)
            hpm = axm.hist(all_priors, bins=bins_hist*2, range=[minmantle,maxmantle], histtype = "step", color='k', alpha=0.8, density=True)
            hppm = axm.hist(all_interfaces, bins=bins_hist*2, range=[minmantle,maxmantle], color=self.hist_col, alpha=0.4, zorder=0, 
                                        histtype="bar",edgecolor='w',linewidth=0.5, density=True)
            
            hmpc = axc.hist(all_mixed_priors, bins=bins_hist*2, range=[mincrust,maxcrust], histtype = "step", color="navy", alpha=0.8, density=True)
            hmpm = axm.hist(all_mixed_priors, bins=bins_hist*2, range=[minmantle,maxmantle], histtype = "step", color="navy", alpha=0.8, density=True)

            if do_MAP: 
                for al in wr_hl:
                    axc.axvline(map_model_depth[al], color=self.map_col, linewidth =1.5, ls='--')
                    axm.axvline(map_model_depth[al], color=self.map_col, linewidth =1.5, ls='--')

            axm.hist([], 40, range=(0,1), label="Posterior", color=self.hist_col, alpha=0.4, zorder=0, histtype="bar",edgecolor='w',linewidth=0.5)
            axm.plot([],[],color='k', ls='-', label="Prior", alpha=0.8)
            if do_MAP: 
                axm.axvline(-1e6, color=self.map_col, linewidth = 1.5, ls='--', label="MAP")
            ### Axis limits 
            axc.set_ylim(0,hppc[0].max())
            axm.set_ylim(0,max(hpm[0].max(), hppm[0].max()))
            axc.set_xlim(-0.3,maxcrust)
            axm.set_xlim(0.99*minmantle,maxmantle)
            axm.get_yaxis().set_ticks([])
            axc.get_yaxis().set_ticks([])
            axc.spines['top'].set_visible(False)
            axc.spines['right'].set_visible(False)
            axc.spines['left'].set_visible(False)
            axm.spines['top'].set_visible(False)
            axm.spines['right'].set_visible(False)
            axm.spines['left'].set_visible(False)
            ###
            axm.set_xlabel("Interface depth [$km$]")
            
            plt.setp(axm.get_xticklabels(), rotation=45, ha='right')
            plt.setp(axc.get_xticklabels(), rotation=45, ha='right')
            leg = axm.legend(loc=1, edgecolor="none", framealpha=0)#.5)
            leg._legend_box.align = "right"
            
            fig2.align_labels()
            fig2.tight_layout()
            if do_save:
                tit= self.method + self.run_name
                if not find_limits: 
                    tit += '_prior'
                fig2.savefig("./Figures/" + tit + "_hist_depth.png".format(u), dpi=600)
                fig2.savefig("./Figures/" + tit + "_hist_depth.pdf".format(u))
        return


    ##########################################################################################################  
    ### MAKE INDIVIDUAL 2D MARGINAL PLOTS 
    ##########################################################################################################  
    def parameter_marginals(self, pars=[[0,1],[0,2],[1,8],[2,8],[5,8],[4,14],[0,8]], find_limits=True, 
                            do_plot_contours=False, do_save=False, do_MAP=True, do_best=False, do_true=False):

        
        ### COLOR PROFILE FOR FIGURE 
        #############################################################################################################
        if not hasattr(self, "true_col"):
            self.color_palette(color_code=1)

        Nparam =  self.flat_samples.shape[1]

        ### Custom limits to compare two inversions 
        # limsc_save = [[[-30.0, 30.0], [53.6, 64.7]], 
        #               [[-30.0, 30.0], [-165.4, -156.2]], 
        #               [[53.6, 64.7], [3.7, 6.1]],
        #               [[-165.4, -156.2], [3.7, 6.1]],
        #               [[1.9, 4.3], [3.7, 6.1]],
        #               [[0.5, 4.0], [0.2, 5.0]],
        #               [[-30.0, 30.0], [3.7, 6.1]]
        #               ]

        ####################################################################################################
        ### Load MAP solution
        if do_MAP: 
            if not hasattr(self, 'map_estimate'):
                self.get_MAP(NM=int(2e4), method="Meanshift") 

        ####################################################################################################
        ### Find Max Likelihood solution
        if do_best:
            idx_best = np.argmax(self.log_flat_samples)

        ####################################################################################################
        ### Plot parameters 
        my_cmap = sns.color_palette("crest", as_cmap=True)
        ### changes the fontsize of matplotlib, not just a single figure
        fontsize=12
        rcParams.update({'font.size': fontsize})

        ####################################################################################################
        ### Loop on parameter pairs 
        for ip, p in enumerate(pars):
            ### PREPARE PLOT 
            u,v = p[0],p[1]
            if find_limits:
                # limsc = limsc_save[ip] 
                limsc = [] 
            else: 
                limsc = self.prior_inv

            qvalues = np.percentile(self.flat_samples[:, u], [16, 84])
            ############################################################
            if find_limits:
                ### FOR U 
                qhist = np.percentile(self.flat_samples[:, u], [1, 99])
                qmin = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
                qmax = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
                ### If the spread is large, just use the priors.
                if abs(qmax-qmin) >= abs(self.prior_inv[u][1] - self.prior_inv[u][0])/1.5:
                    limsc.append(self.prior_inv[u])
                else:
                    limsc.append([qmin, qmax])

                ### FOR V 
                qhist = np.percentile(self.flat_samples[:, v], [1, 99])
                qmin = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
                qmax = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
                ### If the spread is large, just use the priors.
                if abs(qmax-qmin) >= abs(self.prior_inv[v][1] - self.prior_inv[v][0])/1.5:
                    limsc.append(self.prior_inv[v])
                else:
                    limsc.append([qmin, qmax])
            
            ############################################################
            ### Make the 2d histograms
            ### Method for 2D histograms: seaborn plot with corner 
            ### PB: No control of extent of both histograms
            jointgrid = sns.jointplot(x=self.flat_samples[:, v], y=self.flat_samples[:, u],  cmap=my_cmap, kind='hex', 
                                    joint_kws={'mincnt':2, 'extent':[limsc[1][0], limsc[1][1], limsc[0][0], limsc[0][1]], 'gridsize':50 },
                                    marginal_kws={'bins':40,'color':self.hist_col, 'alpha':self.hist_alpha, 'edgecolor':'w','linewidth':0.3})

            fig = jointgrid.figure
            ax = fig.get_axes()[0]
            ### True model point 
            if do_true:
                ax.plot(self.DATA.truth[v],self.DATA.truth[u],  marker='^', markersize = 12, 
                                                c=self.true_col, markeredgecolor="w", markeredgewidth=2, label="True", ls='')
            if do_MAP:
                ax.plot(self.map_estimate[v], self.map_estimate[u], marker='o', markersize = 12, 
                                            c=self.map_col, markeredgecolor="w", markeredgewidth=2, label="MAP", ls='')
            if do_best:
                ax.plot(self.flat_samples[idx_best,v], self.flat_samples[idx_best,u], marker='s', markersize = 12, 
                                            fillstyle='none', markeredgecolor=self.max_col, markeredgewidth=3, label="ML", ls='')
            
            
            ############################################################
            if do_plot_contours:
                plot_contour(ax, Nparam, self.flat_samples[:, v], self.flat_samples[:, u], limsc )
                
            ############################################################
            ### Axis decoration 
            ax.set_xlim(limsc[1])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_xlabel(self.names[v].replace(' [', '\n['))
                
            ax.set_ylim(limsc[0])
            plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
            ax.set_ylabel(self.names[u].replace(' [', '\n['))
                
            ax.locator_params(nbins=5)
            ax.tick_params(axis='both', which='both', labelsize=0.8*fontsize)
            ax.set_facecolor('whitesmoke') 
                
            ### ADD LEGEND:
            ax.legend(loc=0, framealpha=0.5, edgecolor="none")

            ############################################################
            print("Limits = [" + ", ".join("[{:.1f}, {:.1f}]".format(l[0],l[1]) for l in limsc) + "]")
            fig.tight_layout()
            fig.align_labels()
            if do_save:
                tit = self.method + self.run_name
                fig.savefig("./Figures/" + tit + "_hist2d_{:02d}_{:02d}.png".format(u,v), dpi=600)
                fig.savefig("./Figures/" + tit + "_hist2d_{:02d}_{:02d}.pdf".format(u,v))

        return()



    ##########################################################################################################  
    ### MAKE HISTOGRAM CORNER PLOT
    ##########################################################################################################  
    def corner_plot(self, figsize=(10,10), bins_hist=20, find_limits=True, do_plot_contours=False, 
                        do_save=False, do_MAP = True, do_true = False, do_best=False):   

        if find_limits: 
            limsc = [] 
        else: 
            limsc = self.prior_lims
        ### COLOR PROFILE FOR FIGURE 
        #############################################################################################################
        if not hasattr(self, "true_col"):
            self.color_palette(color_code=1)

        if not self.solutions_postprocessed:
            raise TypeError('Please postprocess solutions with .process_solutions()')

        ####################################################################################################
        ### Get Model information 
        Nparam =  self.flat_samples.shape[1]

        ####################################################################################################
        ### Load true solution
        if do_true: 
            truth_inverted = self.DATA.truth[self.any_inverted]

        ####################################################################################################
        ### Find best solution
        if do_best: 
            idx_best = np.argmax(self.log_flat_samples)
        
        ####################################################################################################
        ### Find MAP solution
        if do_MAP: 
            if not hasattr(self, 'map_estimate'):
                self.get_MAP(NM=min(self.n_end, int(2e4)), method="Meanshift") 

        ####################################################################################################
        ### PREPARE PLOT 
        fig = plt.figure(figsize=figsize)
        my_cmap = sns.color_palette("crest", as_cmap=True)
        ### changes the fontsize of matplotlib, not just a single figure
        fontsize=8/10*figsize[0]
        rcParams.update({'font.size': fontsize})
        ###
        grid = fig.add_gridspec(Nparam, Nparam)
        ### Axes for seismic model results 
        axes = [[ 0 for v in range(Nparam)] for u in range(Nparam)]
        ### Axes and share x and y  
        axes[-1][0] = fig.add_subplot(grid[-1,0])
        for u in range(Nparam-1):
            axes[u][0] = fig.add_subplot(grid[u,0],sharex=axes[-1][0])
        for v in range(1,Nparam-1):
            axes[-1][v] = fig.add_subplot(grid[-1,v],sharey=axes[-1][0])
        axes[-1][-1] = fig.add_subplot(grid[-1,-1])
        for u in range(Nparam-1):
            for v in range(1, Nparam):
                if v<u:
                    axes[u][v] = fig.add_subplot(grid[u,v], sharex=axes[-1][v], sharey =axes[u][0])
                elif v==u :
                    axes[u][v] = fig.add_subplot(grid[u,v], sharex=axes[-1][v])


        ####################################################################################################
        ### MAKE PLOT 
        for u in tqdm(range(Nparam), ncols=80, leave=True, position=0, file=sys.stdout):
            qvalues = np.percentile(self.flat_samples[:, u], [16, 84])
            if find_limits:
                qhist = np.percentile(self.flat_samples[:, u], [1, 99])
                qmin = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
                qmax = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
                ### If the spread is large, just use the priors.
                if abs(qmax-qmin) >= abs(self.prior_inv[u][1] - self.prior_inv[u][0])/1.5:
                    limsc.append(self.prior_inv[u])
                else:
                    limsc.append([qmin, qmax])
            for v in range(Nparam):
                if u > v:
                    ### Make the 2d histogram 
                    axes[u][v].hexbin(self.flat_samples[:, v], self.flat_samples[:, u], cmap=my_cmap,
                                            extent=[limsc[v][0], limsc[v][1], limsc[u][0], limsc[u][1]],
                                            mincnt=2, rasterized = True, linewidths=0.0, gridsize=25 )
                    ### True model point 
                    if do_true:
                        axes[u][v].plot(self.DATA.truth[v],truth_inverted[u],  marker='^', markersize = 4*np.sqrt(Nparam/8), 
                                                        c=self.true_col, markeredgecolor=self.true_col, markeredgewidth=np.sqrt(Nparam/6), label=" ", ls='')
                    if do_best:
                        axes[u][v].plot(self.flat_samples[idx_best,v], self.flat_samples[idx_best,u], marker='s', markersize = 4*np.sqrt(Nparam/8), 
                                                        fillstyle='none', markeredgecolor=self.max_col, markeredgewidth=np.sqrt(Nparam/6), label=" ", ls='')
                    if do_MAP: 
                        axes[u][v].plot(self.map_estimate[v], self.map_estimate[u], marker='o', markersize = 4*np.sqrt(Nparam/8), 
                                                        fillstyle='none', markeredgecolor=self.map_col, markeredgewidth=np.sqrt(Nparam/6), label=" ", ls='')
                    if do_plot_contours:
                        plot_contour(axes[u][v], Nparam, self.flat_samples[:, v], self.flat_samples[:, u], 
                                       [[limsc[v][0], limsc[v][1]], [limsc[u][0], limsc[u][1]]] )
                
                elif u==v : 
                    ### Make the 1d histogram 
                    ha = axes[u][u].hist(self.flat_samples[:, u], bins_hist, range=(limsc[u][0],limsc[u][1]),
                                            color=self.hist_col, alpha=self.hist_alpha, zorder=0, histtype="bar",edgecolor='w',linewidth=0.3)
                    if do_true:
                        axes[u][u].axvline(truth_inverted[u], color=self.true_col, linewidth = np.sqrt(Nparam/5), label="True")
                    if do_best:
                        axes[u][u].axvline(self.flat_samples[idx_best,u], color=self.max_col, linewidth = np.sqrt(Nparam/5), ls=':', label="ML")
                    if do_MAP:
                        axes[u][u].axvline(self.map_estimate[u], color=self.map_col, linewidth = np.sqrt(Nparam/5), ls=':', label="MAP")


                ############################################################
                ### Axis decoration 
                if u == Nparam-1:
                    axes[u][v].set_xlim(limsc[v][0], limsc[v][1])
                    plt.setp(axes[u][v].get_xticklabels(), rotation=45, ha='right')
                    axes[u][v].set_xlabel(self.names[v].replace(' [', '\n['))
                if v==0 and u>0:
                    axes[u][v].set_ylim(limsc[u][0], limsc[u][1])
                    plt.setp(axes[u][v].get_yticklabels(), rotation=45, ha='right')
                    axes[u][v].set_ylabel(self.names[u].replace(' [', '\n['))
                
                if u == v  :
                    ### Remove most spines (makes figure less heavy)
                    axes[u][v].spines[['top','left', 'right']].set_visible(False)
                    axes[u][v].yaxis.set_tick_params(labelleft=False, left=False)

                if u<Nparam-1 and v<=u :
                    axes[u][v].xaxis.set_tick_params(labelbottom=False)
                if v>0 and v<u :
                    axes[u][v].yaxis.set_tick_params(labelleft=False)
                if v<=u:
                    if v!=u:
                        axes[u][v].locator_params(nbins=3)
                    axes[u][v].tick_params(axis='both', which='both', labelsize=0.8*fontsize)
                if v<u:    
                    axes[u][v].set_facecolor('whitesmoke') 
                    
        ### ADD LEGEND:
        if do_true:
            axes[1][0].axvline(-1e9, color=self.true_col, linewidth = np.sqrt(Nparam/5), label="True")
        if do_best:
            axes[1][0].axvline(-1e9, color=self.max_col, linewidth = np.sqrt(Nparam/5), ls=':', label="ML")
        if do_MAP:
            axes[1][0].axvline(-1e9, color=self.map_col, linewidth = np.sqrt(Nparam/5), ls=':', label="MAP")
        axes[1][0].legend(loc='center right', bbox_to_anchor=(3.4, 1.6), framealpha=0, ncol=2, columnspacing=-0.6)

        
        print("Limits = [" + ", ".join("[{:.1f}, {:.1f}]".format(l[0],l[1]) for l in limsc) + "]")
        fig.subplots_adjust(wspace=0.12*np.sqrt(4/Nparam), hspace=0.12*np.sqrt(4/Nparam),top=1-0.05*np.sqrt(4/Nparam),
                            bottom=0.2*np.sqrt(4/Nparam), right=1-0.05*np.sqrt(4/Nparam), left=0.2*np.sqrt(4/Nparam))
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            if not find_limits: 
                tit += '_prior'
            fig.savefig("./Figures/" + tit + "_hist2d.png", dpi=600)
            fig.savefig("./Figures/" + tit + "_hist2d.pdf")



    ##########################################################################################################  
    ### MAKE AUTOCORRELATION AND CONVERGENCE ANALYSIS
    ##########################################################################################################  
    def visualize_convergence(self, figsize=(5,4), discard=1000,N_point_autocorr = 50, param = 0, thin=1, do_save=False):

        opt = dict(discard=discard, thin=thin, flat=False)
        print("Loading data...")
        if self.method=="emcee":
            self.samples = self.sampler.get_chain(**opt)[:,:]
        else:
            with h5py.File(self.filename, "r") as f:
                chain = f["chain"][:]  
                self.samples = self._get_value(chain, **opt)
        
        N_select= self.samples.shape[1]
        taus = np.zeros((N_select,N_point_autocorr ))
        lasts = []
        print("Calculating autocorrelation times...")
        for j in tqdm(range(N_select),ncols=80, leave=True, position=0, file=sys.stdout):
            res_j = self.samples[:,j,param]   # select one emcee ensemble
            last = np.size(res_j)
            if discard>last :
                discard = 100
            if discard==0:
                discard=1
            lims = 10**np.linspace(np.log10(discard),np.log10(last),N_point_autocorr)
            lasts.append(lims)
            for ll,lim in enumerate(lims) :
                lim = int(lim)
                taus[j,ll] = autocorr_new(res_j[:lim])

        ##############################################################
        # Make plots of ACF estimate for a few different chain lengths
        fontsize=12/5*figsize[0]
        rcParams.update({'font.size': fontsize})
        fig, axes = plt.subplots( figsize=figsize)
        for j in range(N_select):
            axes.plot(lasts[j], taus[j,:], 'o-', c='k',lw=0.8, alpha=0.5, markersize=0.8)
        # Can be done only if all lims are the same, i.e simulation is finished
        try:
            axes.plot(lasts[-1], np.mean(taus, axis=0), color='darkturquoise',lw=3)
        except: 
            pass
        # Plot when we have reached 50 uncorelated samples  
        milast, malast = np.min(np.array(lasts)),np.max(np.array(lasts))
        ait = 10**np.linspace(np.log10(milast),np.log10(malast),N_point_autocorr)
        axes.plot(ait, ait / 50, "--b", label=r"$\tau = N_{iterations}/50$")
        axes.plot([],[],'o-', c='k',lw=1, alpha=0.5, markersize=1, label=r"chains")
        axes.set_xlabel(r"$N_{iterations}$")
        axes.set_ylabel(r"Estimated $\tau_c$" +r" (from " + self.names[param].split(' [')[0] + ")")
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlim(milast,malast+thin+discard)
        axes.grid(ls=':')
        axes.legend(loc=2,framealpha=0, edgecolor='none')
        ###
        fig.tight_layout() 
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig.savefig("./Figures/" + tit + "_convergence_{:02d}.png".format(param), dpi=600)
            fig.savefig("./Figures/" + tit + "_convergence_{:02d}.pdf".format(param))


    ##########################################################################################################  
    ### INSPECT CHAIN 
    ##########################################################################################################  
    def inspect_chain(self, figsize=(10,4), param=0, direction="horizontal", thin=1, do_save=False, irun=0):
        ### Code acts differently for emcee and ptmcmc. 
        ### EMCEE: plot all chains of the ensemble
        ### PTMCMC: plot all temperatures for one cpu

        ### Plot some parameters out of the chains, the acceptance rate and the misfit
        ### And their variations with iterations.  
        print("Loading data...")

        n_chain = max(self.n_temp, self.n_walk)
        NM = self.n_dim

        ### Shape [N_iter, N_chain, N_param]
        if self.method=="emcee":
            self.samples = self.sampler.get_chain(thin=thin)
            self.log_samples = self.sampler.get_log_prob(thin=thin)
            #blobs = self.sampler.get_blobs()

            n_end = self.sampler.iteration
            iterations = np.linspace(1,n_end, self.log_samples.shape[0]) 
            ### Load and downsample acceptance rate 
            self.acc_list = np.load(self.save_dir + 'acceptance_rate.npy')[:n_end][::thin]
            ### If the chain is still running, 
            ### acceptance chain may be a little 
            ### smaller due to the gap between saves 
            iterations_acc = np.linspace(1,n_end, self.acc_list.shape[0])

            ### For autocorrelation calculation, there should be no thining. 
            chain_autocorr = self.sampler.get_chain()[:,:,param]

        elif self.method=="emcee2":
            with h5py.File(self.filename, "r") as f:
                chain = f["chain"][:]  # Shape: (nsteps, nwalkers, ndim)
                log_prob = f["log_prob"][:]  # Shape: (nsteps, nwalkers)
                accepted = f["accepted"][:]  # Shape: (nsteps,nwalkers)
                acceptance_rate = accepted/np.arange(1,accepted.shape[0]+1)[:,None]

                self.samples = self._get_value(chain, thin=thin)
                self.log_samples = self._get_value(log_prob, thin=thin)
                self.acc_list = self._get_value(acceptance_rate, thin=thin)

                n_end = self.samples.shape[0]
                iterations = np.linspace(1,n_end, self.log_samples.shape[0]) 
                iterations_acc = iterations

                ### For autocorrelation calculation, there should be no thining. 
                chain_autocorr = chain[:,:,param]

        elif self.method=="ptmcmc":      
            print(self.filenames)
            with h5py.File(self.filenames[irun], "r") as f:
                chain = f["chain"][:]  # Shape: (nsteps, nwalkers, ndim)
                log_prob = f["log_prob"][:]  # Shape: (nsteps, nwalkers)
                accepted = f["accepted"][:]  # Shape: (nsteps,nwalkers)
                acceptance_rate = accepted/np.arange(1,accepted.shape[0]+1)[:,None]

                self.samples = self._get_value(chain, thin=thin)
                self.log_samples = self._get_value(log_prob, thin=thin)
                self.acc_list = self._get_value(acceptance_rate, thin=thin)

                n_end = self.samples.shape[0]
                iterations = np.linspace(1,n_end, self.log_samples.shape[0]) 
                iterations_acc = iterations

                ### For autocorrelation calculation, there should be no thining. 
                chain_autocorr = chain[:,:,param] 
            
        ### Choose the variable for histogram, select all ensembles
        choice_chain = self.samples[:,:,param]    

        ######################################################################
        ### Compute autocorrelation function
        ### For ptmcmc: calculate only for cold chain
        if self.method=="ptmcmc":
            taus = np.zeros(20)
            print("Calculating autocorrelation time...")
            res_j = chain_autocorr[:,0]   # select cold chain 
            lims = 10**np.linspace(np.log10(100),np.log10(n_end),taus.size)
            lasts=lims
            for ll,lim in enumerate(lims) :
                lim = int(lim)
                taus[ll] = autocorr_new(res_j[:lim])
        ### For emcee: calculate for all chains 
        else:
            taus = np.zeros((n_chain,20)) 
            lasts = []
            print("Calculating autocorrelation time...")
            for j in tqdm(range(n_chain),ncols=80, leave=True, position=0, file=sys.stdout):
                res_j = chain_autocorr[:,j]   # select one emcee ensemble
                lims = 10**np.linspace(np.log10(100),np.log10(n_end),taus.shape[1])
                lasts.append(lims)
                for ll,lim in enumerate(lims) :
                    lim = int(lim)
                    taus[j,ll] = autocorr_new(res_j[:lim])


        ######################################################################
        ### Find current best chain 
        chain_best = np.argwhere(self.log_samples==self.log_samples.max())[0][1]

        if self.method == "emcee" or self.method == "emcee2":
            alpha = np.ones(n_chain)*0.5
            evenly_spaced_interval = np.linspace(0.2, 1, n_chain)
            color = [cm.binary(x) for x in evenly_spaced_interval]
            lws = [0.5 for i in range(n_chain)]
            zos = [1 for i in range(n_chain)]
            color[chain_best] = "crimson"
            alpha[chain_best] = 1.0
            lws[chain_best] = 1.2
            zos[chain_best] = 100
        elif self.method=="ptmcmc" :
            evenly_spaced_interval = np.linspace(0.2, 1, n_chain-1)
            ### Cold chain is the first chain 
            color = ['crimson'] + [cm.binary_r(x) for x in evenly_spaced_interval]
            lws = [1] + [0.5 for i in range(n_chain-1)]
            zos = [100] + [1 for i in range(n_chain-1)]
            alpha = [1] + [0.5 for i in range(n_chain-1)]


        ######################################################################
        ### Plot evolution of misfit 
        fig1, ax1 = plt.subplots(figsize=(8,4))
        print("Displaying misfit...")
        
        if self.method == "ptmcmc":
            for i in range(n_chain):
                ax1.plot(iterations, -self.log_samples[:,i], lw=lws[i], color=color[i], zorder=zos[i], label = "T={:.0f}".format(self.Tladder[i]))
        else:
            for i in tqdm(range(n_chain),ncols=80, leave=True, position=0, file=sys.stdout):
                nmod = 0
                ax1.plot(iterations, -self.log_samples[:,i], lw=lws[i], color=color[i], zorder=zos[i])
            ax1.plot([],[],lw=lws[chain_best], color=color[chain_best], label="Best chain")
        ax1.set_ylabel("-Log likelihood")
        ax1.set_xlabel("Iterations")
        ax1.legend(loc=1, framealpha=1, edgecolor='None')
        ax1.set_yscale("log")
        ax1.grid(ls=':')
        ax1.set_xlim(0,n_end-1)
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        fig1.tight_layout()

        ######################################################################
        ### Plot acceptance rate evolution
        fig2, ax2 = plt.subplots(figsize=(8,4))
        print("Displaying acceptance rate...")
        if self.method == "ptmcmc":
            for i in range(n_chain):
                ax2.plot(iterations_acc, self.acc_list[:,i], lw=lws[i], color=color[i], zorder=zos[i], label = "T={:.0f}".format(self.Tladder[i]))
            #ax2.plot([],[],lw=lws[0], color=color[0], label="Cold chain")
        else:
            for i in tqdm(range(n_chain),ncols=80, leave=True, position=0, file=sys.stdout):
                nmod = 0
                # ax2.plot(iterations_acc, self.acc_list[:,-i-1], lw=lws[-i-1], color=color[-i-1], zorder=zos[-i-1])
                ax2.plot(iterations_acc, self.acc_list[:,i], lw=lws[i], color=color[i], zorder=zos[i])
            ax2.plot([],[],lw=lws[chain_best], color=color[chain_best], label="Best chain")
        ax2.set_ylabel("Acceptance rate")
        ax2.set_xlabel("Iterations")
        ax2.legend(loc=1, framealpha=1, edgecolor='None')
        ax2.grid(ls=':')
        ax2.set_ylim(0,0.5)
        ax2.set_xlim(0,n_end-1)
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        fig2.tight_layout()


        ######################################################################
        ### We are going to plot in a 2 by X grid (horizontal or vertical:
        ### - chains for each parameter on a 2 by X grid
        ### - acceptance of temperature chains
        ### - acceptance of swap
        ### - autocorrelation time
        ### - because autocorrelation time is computed with the first param,
        ###   we can also plot the histogram of this param : see if it varies between chains.
        Nshort = 2
        Nlong = (NM//Nshort + NM%Nshort) + 2
        if direction=="horizontal":
            fig, ax = plt.subplots(Nshort,Nlong, figsize = (2*Nlong,2*Nshort))
        else:
            fig, ax = plt.subplots(Nlong,Nshort, figsize = (2*Nshort,2*Nlong))
        ########################################################
        print("Displaying chains...")
        for i in tqdm(range(n_chain),ncols=80, leave=True, position=0, file=sys.stdout):
            nmod = 0
            for ilong in range(Nlong) :
                for ishort in range(Nshort):
                    if direction=="horizontal":
                        u=ishort 
                        v=ilong 
                    else:
                        u=ilong
                        v=ishort
                    if ilong<(NM//Nshort + NM%Nshort):
                        if nmod < NM :
                            ax[u][v].plot(iterations, self.samples[:,i,nmod],
                                            alpha=alpha[i], lw=lws[i], color=color[i], zorder=zos[i])
                            nmod+=1
                        else :
                            ax[u][v].axis('off')
                    else :
                        if direction=="horizontal":
                            ### Acceptance evolution
                            ax[0][Nlong-2].plot(iterations_acc, self.acc_list[:,i], lw=lws[i], 
                                                color=color[i], zorder=zos[i], alpha=alpha[i])
                            ### Misfit evolution
                            ax[1][Nlong-2].plot(iterations,-self.log_samples[:,i], lw=lws[i], 
                                                color=color[i], zorder=zos[i], alpha=alpha[i])
                            ### Current autocorrelation
                            if self.method =="emcee":
                                ax[0][Nlong-1].plot(lasts[i],taus[i,:], marker = 's', markersize=0.6,
                                                        lw=lws[i],c=color[i],alpha=alpha[i], zorder=zos[i])
                            elif self.method == "ptmcmc" and i==0:
                                ax[0][Nlong-1].plot(lasts,taus, 'k-', marker = 's', markersize=0.6, lw=0.5)
                            ### Histogram of best chain 
                            ax[1][Nlong-1].hist(choice_chain[:,chain_best],bins=20, 
                                                range=(self.prior_lims[param][0],self.prior_lims[param][1]), 
                                                color="crimson", alpha=0.4, histtype="bar",edgecolor='white',linewidth=0.3)
                        else:
                            ### Acceptance evolution
                            ax[Nlong-2][0].plot(iterations_acc, self.acc_list[:,i], lw=lws[i], 
                                                color=color[i], zorder=zos[i], alpha=alpha[i])
                            ### Misfit evolution
                            ax[Nlong-2][1].plot(iterations,-self.log_samples[:,i], lw=lws[i], 
                                                color=color[i], zorder=zos[i], alpha=alpha[i])
                            ### Current autocorrelation
                            if self.method =="emcee":
                                ax[Nlong-1][0].plot(lasts[i],taus[i,:], marker = 's', markersize=0.6,
                                                        lw=lws[i],c=color[i],alpha=alpha[i], zorder=zos[i])
                            elif self.method == "ptmcmc" and i==0:
                                ax[Nlong-1][0].plot(lasts,taus, 'k-', marker = 's', markersize=0.6, lw=0.5)
                            ### Histogram of one parameter 
                            ax[Nlong-1][1].hist(choice_chain[:,chain_best],bins=20, 
                                                range=(self.prior_lims[param][0],self.prior_lims[param][1]), 
                                                color="crimson", alpha=0.4, histtype="bar",edgecolor='white',linewidth=0.3)
        
        ####################################
        ### Axis decoration 
        nmod = 0
        for ilong in range(Nlong) :
            for ishort in range(Nshort):
                if direction=="horizontal":
                    u=ishort 
                    v=ilong 
                else:
                    u=ilong
                    v=ishort
                if u+v != Nlong+Nshort-2 and ilong!=Nlong-1:
                    ax[u][v].set_xlim([0,n_end-1])
                    ax[u][v].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
                if ilong<(NM//Nshort + NM%Nshort):
                    if nmod < NM :
                        ax[u][v].set_ylabel(self.names[nmod])
                        ax[u][v].set_ylim(self.prior_inv[nmod][0],self.prior_inv[nmod][1])
                        if ishort==1:
                            ax[u][v].set_xlabel("Iteration")
                        ax[u][v].set_xlim([0,n_end-1])
                        ax[u][v].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
                        nmod+=1
                else :
                    if direction=="horizontal":
                        ax[0][Nlong-2].set_ylim([0,1])
                        # ax[1][Nlong-2].set_ylim([0,1])
                        ax[0][Nlong-2].set_ylabel(r"Accept. rate")
                        ax[0][Nlong-1].set_ylabel(r"Estimated $\tau_m$")
                        ax[0][Nlong-1].set_xscale("log")
                        ax[0][Nlong-1].set_yscale("log")
                        ax[0][Nlong-1].grid(ls=':',c='k')
                        ax[1][Nlong-1].set_xlabel(self.names[0])
                        ax[1][Nlong-1].set_ylabel(r"Count of "+self.names[param])
                        ax[0][Nlong-1].set_xlabel("Iteration")
                        ax[1][Nlong-1].set_xlabel("Iteration")
                        ax[1][Nlong-2].set_yscale("log")
                        ax[1][Nlong-2].set_ylabel(r"- Log-Likelihood")
                        ax[1][Nlong-2].set_xlabel("Iteration")
                    else:
                        ax[Nlong-2][0].set_ylim([0,1])
                        # ax[Nlong-2][1].set_ylim([0,1])
                        ax[Nlong-2][0].set_ylabel(r"Accept. rate")
                        ax[Nlong-1][0].set_ylabel(r"Estimated $\tau_m$")
                        ax[Nlong-1][0].set_xscale("log")
                        ax[Nlong-1][0].set_yscale("log")
                        ax[Nlong-1][0].grid(ls=':',c='k')
                        ax[Nlong-1][1].set_xlabel(self.names[0])
                        ax[Nlong-1][1].set_ylabel(r"Count of "+self.names[param])
                        ax[Nlong-1][0].set_xlabel("Iteration")
                        ax[Nlong-1][1].set_xlabel("Iteration")
                        ax[Nlong-2][1].set_ylabel(r"- Log-Likelihood")
                        ax[Nlong-2][1].set_yscale("log")
                        ax[Nlong-2][1].set_xlabel("Iteration")                        
        ###
        fig.tight_layout()
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig.savefig("./Figures/" + tit + "_inspect.png", dpi=600)
            fig.savefig("./Figures/" + tit + "_inspect.pdf")
            fig1.savefig("./Figures/" + tit + "_misfit.png", dpi=600)
            fig1.savefig("./Figures/" + tit + "_misfit.pdf")
            fig2.savefig("./Figures/" + tit + "_acceptance.png", dpi=600)
            fig2.savefig("./Figures/" + tit + "_acceptance.pdf")


    ##########################################################################################################  
    ### MAKE LOCATION MAP 
    ##########################################################################################################  
    def source_location(self, figsize=(4,4), do_save=False, geography = False, zoom=False):

        from mpl_toolkits.basemap import Basemap

        wr_Ls = self.wr_order[1]

        ### Get Posterior data and make 2D hist 
        hlat, hlon = self.flat_samples[:,wr_Ls[0]], self.flat_samples[:,wr_Ls[1]]
        ### Find limits 
        qhist = np.percentile(self.flat_samples[:, wr_Ls[1]], [1, 99])
        qmin_lon = qhist[0] - 0.1*abs(qhist[1]-qhist[0])
        qmax_lon = qhist[1] + 0.1*abs(qhist[1]-qhist[0])
        ###
        qhist = np.percentile(self.flat_samples[:, wr_Ls[0]], [1, 99])
        qmin_lat = qhist[0] - 0.2*abs(qhist[1]-qhist[0])
        qmax_lat = qhist[1] + 0.2*abs(qhist[1]-qhist[0])

        lats_all = self.DATA.sta_lats + [qmax_lat] + [qmin_lat] 
        lons_all = self.DATA.sta_lons + [qmax_lon] + [qmin_lon] 

        lat_bins = np.linspace(qmin_lat, qmax_lat, 20)
        lon_bins = np.linspace(qmin_lon, qmax_lon, 20)
        # lat_bins = np.linspace(min(lats_all), max(lats_all), 40)
        # lon_bins = np.linspace(min(lons_all), max(lons_all), 40)

        hist, lonedge , latedge = np.histogram2d(hlon, hlat, [lon_bins, lat_bins], 
                                    density = False)
        ### Normalizes histogram by number of points to get probability
        #HL = np.where(hist!=0)[0].size  
        hist[np.where(hist<10)] = np.nan
        #hist/=HL
        hist/=np.nanmax(hist)
        
        lonedge_center = (lonedge[:-1] + lonedge[1:]) / 2
        latedge_center = (latedge[:-1] + latedge[1:]) / 2
        histmax = 1.0

        ####################################################################################################
        ### Load true solution
        _, lat_true, lon_true, source_depth_true = self.DATA.truth_location

        fig, ax = plt.subplots(figsize = figsize)

        if zoom: 
            m = Basemap(ax=ax, projection='lcc', resolution='i', 
                            lat_0=(max(lats_all)+min(lats_all))/2, lon_0=(max(lons_all)+min(lons_all))/2,
                            llcrnrlon= min(lons_all)-5,llcrnrlat=min(lats_all)-5,
                            urcrnrlon=max(lons_all)+10,urcrnrlat=max(lats_all)+5 )
            elat, elon = 5, 10 
        else: 
            m = Basemap(ax = ax, projection='robin',lon_0=0,resolution='i')
            elat, elon = 20, 30

        # Draw coastlines and country borders
        if geography:
            m.drawcoastlines(color="#94568c", linewidth =0.5)
            m.drawmapboundary(fill_color='lavender')
            m.fillcontinents(color='darksalmon',lake_color='lavender')

        # Draw parallels and meridians
        m.drawparallels(np.arange(-90, 90, elat), labels=[1,0,0,0] ,linewidth=0.6, color="grey")
        m.drawmeridians(np.arange(-180, 180, elon), labels=[0,0,0,1] , linewidth=0.6, color="grey")

        lons_cont, lats_cont = np.meshgrid(lonedge_center, latedge_center) # get lat/lons of ny by nx evenly space grid.
        xl, yl = m(lons_cont, lats_cont) # compute map proj coordinates.
        # draw filled contours.
        from matplotlib.colors import ListedColormap
        levels = np.linspace(0,histmax, 21)
        list_cmap = sns.color_palette("rocket", as_cmap=True)
        new_cmap = list_cmap
        # list_cmap = cm.get_cmap('hot')
        # cmap = list_cmap(np.linspace(0.1, 0.8, 256))  # Get RGB values
        # cmap[:,-1] = np.logspace(np.log10(0.3), 0, len(cmap))
        # new_cmap = ListedColormap(cmap)
        cs = m.contourf(xl,yl,hist.T,levels=levels,cmap=new_cmap) #"gist_heat")#, vmin=0, vmax=histmax)
        cbar = m.colorbar(cs,location='bottom',pad="28%", ticks=np.linspace(0,histmax,6)) #ticks=np.linspace(0,histmax, int(histmax//0.1)+1))
        cbar.set_label('Probability density of location (Norm.)')

        # Convert latitude and longitude to x and y coordinates and plot event 
        x_B, y_B = m(self.DATA.ev_lon, self.DATA.ev_lat)
        m.plot(x_B, y_B, markerfacecolor = '#51eaf2', markeredgecolor="k", marker="*", markersize=12)

        # Plot the stations
        for i , (slat, slon) in enumerate(zip(self.DATA.sta_lats, self.DATA.sta_lons)):
            x_A, y_A = m(slon, slat)
            m.plot(x_A, y_A, '#e40e8b', marker="^", markeredgecolor="k", markersize=10, zorder=100)
            ax.text(x_A+1e5, y_A+2e5, '{:d}'.format(i), ha="center", va="top", 
						bbox=dict(alpha=0.4, facecolor='white', pad=0, linewidth=0))

        ### TITLE AND LABELS 
        ax.set_ylabel(r"Latitude / [$^{\circ}]$", labelpad=30)
        ax.set_xlabel(r"Longitude / [$^{\circ}]$", labelpad=15)

        ### LEGEND 
        ax.plot([],[],ls="", markerfacecolor = '#51eaf2', markeredgecolor="k", marker="*", markersize=12, label="True Epicenter")
        if self.DATA.sta_alts[0] == 0:
            balloon_lat = [-2.570, 8.041, 5.985, -5.827]              # approximate latitude during event
            balloon_lon = [125.7318, 121.4965, 143.6585, 143.1594]
            for i, (blat, blon) in enumerate(zip(balloon_lat, balloon_lon)):
                x_A, y_A = m(blon, blat)
                m.plot(x_A, y_A, marker="+", markeredgecolor="k", markersize=10, markeredgewidth=3, alpha=0.5, zorder=102)
            ax.plot([],[],ls="", c='#e40e8b', marker="^", markeredgecolor="k", markersize=10, label="Stations")
        else:
            ax.plot([],[],ls="", c='#e40e8b', marker="^", markeredgecolor="k", markersize=10, label="Balloons")
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.7), framealpha=0, ncol=2)

        ###
        fig.tight_layout()
        fig.align_labels()
        if do_save:
            tit = self.method + self.run_name
            fig.savefig("./Figures/" + tit + "_mapsource.png", dpi=600,bbox_inches='tight')
            fig.savefig("./Figures/" + tit + "_mapsource.pdf",bbox_inches='tight')
