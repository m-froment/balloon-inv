import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from matplotlib import ticker
import pycwt as wavelet
###
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.event import read_events
from obspy import read , Stream, Trace
from obspy.core.trace import Stats
###
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit 
from scipy.signal import resample
from scipy.fft import rfft, rfftfreq



########################################################################################################
def next_power_of_2(x):
	x= int(x)
	return 1 if x == 0 else 2**(x - 1).bit_length()


########################################################################################################
def scientific_10(x, pos):
	if abs(x)==0:
		return r"${: 1.0f}$".format(x)
	else :
		exponent = np.floor(np.log10(abs(x)))
		coeff = x/10**exponent
		if int(coeff) ==1:
			return r"$10^{{ {:.0f} }}$".format(exponent)
		# if abs(exponent)==0:
		#     return r"${: 2.1f} $".format(x)
		# elif abs(exponent)==1:
		#     return r"${: 2.0f} $".format(x)
		# elif exponent ==2:
		#     return r"${: 3.0f} $".format(x)
		else :
			return r"${: 2.0f} \times 10^{{ {:.0f} }}$".format(coeff,exponent)



########################################################################################################
def plot_scalogram(tr, ax, ax_cb=None, title_unit='', font=10, graph="pmesh", whiten=0, 
						fmin=None, fmax=None, cmap="magma", t_origin = None, **kwargs):	

	### Read trace with index itr (zero by default) 
	### Calculate wavelet spectrogram (scalogram) and plots it on ax 
	### Returns ax and possibly colorbar ax_cb 

	dt = tr.stats.delta
	mpl_factor = 3600*24
	if t_origin is not None: 
		mpl_factor = 1

	pad = int(next_power_of_2(tr.stats.npts)-tr.stats.npts)     # Pads the time series with zeroes (recommended)
	dj = 1/16                                                   # Uses 1/dj sub-octaves per octave
	s0 = 2*dt                                                   # Start at a scale of 2*dt 
	fdisp_min = 5e-4                                            # Minimum frequency to display
	noct = int(np.log((1/s0)/fdisp_min)/np.log(2))              # Estimates the number of octaves from min frequency
	J = noct / dj                                               # Number of octave / voices = total number of scales 
	#print(noct, pad)
	mother = 'MORLET'

	#############################################
	if fmin is not None and fmax is not None:
		tr.detrend()
		tr.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
	elif fmin is not None and fmax is None:
		tr.detrend()
		tr.filter('highpass', freq=fmin, zerophase=True)
	elif fmin is None and fmax is not None:
		tr.detrend()
		tr.filter('lowpass', freq=fmax, zerophase=True)
	else:
		tr.detrend()
		fmax = 1/(2*dt)

	#############################################
	np.int = int
	wave, scale, f, coi, fft, fftfreqs = wavelet.cwt(tr.data, dt, dj, s0, J,  wavelet.Morlet())   ### Pycwt version
	p = np.abs(wave)**2  # compute wavelet power spectrum
	# p /= scale[:, None]

	### Select data within fmin, fmax 
	bol = np.array((f > 0, f<1e6)).all(axis=0)
	### Select frequency axis 
	fr = f[bol]
	
	#############################################
	### Set the DB scale
	if whiten==0:
		col = 10 * np.log10(p[bol, :])
		if "dbrange" in kwargs:
			if kwargs["dbrange"][0] == "min": 
				dBmin = np.nanmin(col)
			else:
				dBmin = kwargs["dbrange"][0]
			###
			if kwargs["dbrange"][1] == "max": 
				dBmax = np.nanmax(col)		
			else:
				dBmax = kwargs["dbrange"][1]
			# print(np.nanmin(col),np.nanmax(col))	
		else :
			#dBmin, dBmax = np.nanmin(col)/1.5, np.nanmax(col)
			wrf = np.where((fr>5e-2) & (fr<2e-1))
			dBmin = np.mean(col[wrf]) - 1*np.std(col[wrf])
			dBmax = col[wrf].max()
	elif whiten==1:
		### Whitening the spectrogram by the median signal in time
		pb = p[bol, :]
		### Whitening with median 
		pbm = np.nanmedian(pb,axis=1)
		col = 10 * np.log10(p[bol, :]/pbm[:,np.newaxis])
		dBmin, dBmax = np.nanmin(col)/1.5, np.nanmax(col)
	

	#############################################
	### Convert to matplotlib timestamps or not  
	tc = tr.times("matplotlib")
	### Optionnaly, centers on event origin time 
	if t_origin is not None:
		tc -= t_origin.timestamp/mpl_factor
	
	############################################
	if graph=="pmesh":
		### Option 1: pcolormesh
		ct = ax.pcolormesh(tc, fr, col,rasterized=True,
						vmin= dBmin, vmax=dBmax, cmap=cmap,shading='auto')
		fpmin = fr.min()
	elif graph=="cont":
		### Option 2: contourf
		lev = np.linspace(dBmin, dBmax,40)
		if whiten==0:
			lev = (lev*10)//10
		ct = ax.contourf(tc,fr,col, levels=lev,
					cmap=cmap, extend="both")#,vmin= dBmin, vmax=dBmax)#, extend="both")
		fpmin = fr.min()
		for c in ct.collections:
			c.set_rasterized(True)

	#################################################################
	### Main axis decorations 
	ax.set_yscale('log')
	ax.set_ylabel('Frequency / $Hz$', fontsize=font)
	ax.set_ylim([fpmin, fmax])
	ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
	if t_origin is None:
		ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
		# Rotates and right-aligns the x labels so they don't crowd each other.
		for label in ax.get_xticklabels(which='major'):
			label.set(rotation=30, horizontalalignment='right')
	ax.tick_params(axis='both', which='both', labelsize=font-2)

	#################################################################
	### Region of significance 
	# coi[np.where(coi==0)] = 1e-15
	### Wavelets
	# ax.fill_between(tc, 1/(coi * 0 + 1e8), [max(1/coii,1/period[-1]) for coii in coi ], 
	# 					facecolor="k", edgecolor="none", alpha=0.4)
	#print(coi, 1/coi)
	#ax.plot(tc, 2/coi, 'k')

	### COLORBAR ####################################################
	if ax_cb is not None:
		cb = plt.colorbar(mappable=ct, cax=ax_cb)
		if whiten == 0:
			ax_cb.set_ylabel(r'PSD / $dB$$\cdot$$Hz^{-1}$', fontsize=font)
		elif whiten == 1: 
			ax_cb.set_ylabel(r'Whitened PSD', fontsize=font)
		ax_cb.tick_params(axis='both', which='both', labelsize=font-2)
		ax_cb.yaxis.set_label_position('left')
		ax_cb.set_title(title_unit ,fontsize=font)
	
	return(ax, ax_cb)
########################################################################################################


########################################################################################################
### STEP ONE: INITIALIZE STREAMS 
class BALLOON_DATA():

	def __init__(self):
		### Get Flores event metadata
		ddir = "./Flores_data/"
		catalog = read_events(ddir + "flores_catalog_entry.xml", format="QUAKEML")

		self.lat_e = catalog[0].origins[0].latitude
		self.lon_e = catalog[0].origins[0].longitude
		self.dep_e = catalog[0].origins[0].depth
		self.t_event = catalog[0].origins[0].time 
		self.m_e = catalog[0].magnitudes[0].mag
		print("Magnitude: ", self.m_e) 
		print("Time: ", self.t_event.strftime("%d-%m-%y"))

		##############################################################################################################################################
		### OPENING AND PLOTTING FLORES BALLOON DATA 
		##############################################################################################################################################
		### Select directory for the data in .mseed format 
		seedlist = []
		### Loop on balloons
		files = [["Pressure_traces_TTL3_17.mseed","Altitude_traces_TTL3_17.mseed", "Latitude_traces_TTL3_17.mseed", "Longitude_traces_TTL3_17.mseed"],
		   		 ["Pressure_traces_TTL5_16.mseed","Altitude_traces_TTL5_16.mseed", "Latitude_traces_TTL5_16.mseed", "Longitude_traces_TTL5_16.mseed"],
				 ["Pressure_traces_TTL4_15.mseed","Altitude_traces_TTL4_15.mseed", "Latitude_traces_TTL4_15.mseed", "Longitude_traces_TTL4_15.mseed"],
		   		 ["Pressure_traces_TTL4_07.mseed","Altitude_traces_TTL4_07.mseed", "Latitude_traces_TTL4_07.mseed", "Longitude_traces_TTL4_07.mseed"]]
		for bal in files:
			chan_list = []
			### Loop on channels:
			for f in bal:
				### Return read seed. 
				chan_list.append(read(ddir + f).merge())
			seedlist.append(chan_list)

		### Assign
		self.seeds = seedlist 

		### Find out units 
		unitlist = []
		self.id_pre = 0 
		self.id_alt = 1 
		self.id_lat = 2 
		self.id_lon = 3 
		self.id_tem = 4
		for bal in self.seeds:
			chan_unit = []
			for si, s in enumerate(bal) : 
				if s[0].stats.channel == "pre" or  s[0].stats.channel == "pr1": 
					chan_unit.append(r"Pressure / [$Pa$]")
					id_pre = si 
				elif s[0].stats.channel == "alt" : 
					chan_unit.append(r"Altitude / [$m$]")
					id_alt = si 
				elif s[0].stats.channel == "lat" : 
					chan_unit.append(r"Latitude / [$^{\circ}$]")
					id_lat = si 
				elif s[0].stats.channel == "lon" : 
					chan_unit.append(r"Longitude / [$^{\circ}$]")
					id_lon = si 
				elif s[0].stats.channel == "tem" : 
					chan_unit.append(r"Temperature / [$K$]")
					id_tem = si 
			unitlist.append(chan_unit)
		self.units = unitlist


########################################################################################################
### STEP TWO: INTERPOLATE THROUGH GAPS  
def correct_gaps(bdata):
	### Store streams in list: 
	list_streams = []

	### Big loop on each balloon:
	for ibal, sbal in enumerate(bdata.seeds):

		##############################################################################################################################################
		### Step0: interpolate 
		### We interpolate through gaps with the Akima method 
		### It simplifies the treatment and avoids having weird border effects. 

		gaps = sbal[bdata.id_pre].get_gaps() 
		if len(gaps)>0:
			print("Removing gaps for balloon " + sbal[bdata.id_pre][0].stats.station)

			### Copy data with gaps 
			st_p = sbal[bdata.id_pre].copy().split()
			st_z = sbal[bdata.id_alt].copy().split()
			st_lat = sbal[bdata.id_lat].copy().split()
			st_lon = sbal[bdata.id_lon].copy().split()
			dtz = st_z[0].stats.delta
			dtp = st_p[0].stats.delta
			dtlat = st_lat[0].stats.delta
			dtlon = st_lon[0].stats.delta

			### Extract values accounting for gaps 
			times_p = np.concatenate([tr.times("timestamp") for tr in st_p])
			values_p= np.concatenate([tr.data for tr in st_p])
			times_z = np.concatenate([tr.times("timestamp") for tr in st_z])
			values_z= np.concatenate([tr.data for tr in st_z])
			times_lat = np.concatenate([tr.times("timestamp") for tr in st_lat])
			values_lat= np.concatenate([tr.data for tr in st_lat])
			times_lon = np.concatenate([tr.times("timestamp") for tr in st_lon])
			values_lon= np.concatenate([tr.data for tr in st_lon])

			### Time array with no gap
			times_wanted_p = np.arange(st_p[0].times("timestamp")[0], st_p[-1].times("timestamp")[-1]+dtp, dtp)
			times_wanted_z = np.arange(st_z[0].times("timestamp")[0], st_z[-1].times("timestamp")[-1]+dtz, dtz)
			times_wanted_lat = np.arange(st_lat[0].times("timestamp")[0], st_lat[-1].times("timestamp")[-1]+dtlat, dtlat)
			times_wanted_lon = np.arange(st_lon[0].times("timestamp")[0], st_lon[-1].times("timestamp")[-1]+dtlon, dtlon)

			### Replace data with interpolated data 
			sbal[bdata.id_pre][0].data = Akima1DInterpolator(times_p, values_p)(times_wanted_p) 
			sbal[bdata.id_alt][0].data = Akima1DInterpolator(times_z, values_z)(times_wanted_z)
			sbal[bdata.id_lat][0].data = Akima1DInterpolator(times_lat, values_lat)(times_wanted_lat)
			sbal[bdata.id_lon][0].data = Akima1DInterpolator(times_lon, values_lon)(times_wanted_lon)

		st_p = sbal[bdata.id_pre]
		st_z = sbal[bdata.id_alt]
		st_lat = sbal[bdata.id_lat]
		st_lon = sbal[bdata.id_lon]
		list_streams.append([st_p, st_z, st_lat, st_lon])
		print(st_lat, st_lon)
	
	return(list_streams)


########################################################################################################
### STEP THREE: UPSAMPLE Z AND PLOT RELATION WITH P  
def Z_upsample_align(list_streams, do_plot=False):	
	### We will first upsample Z 
	
	### Big loop on each balloon:
	for ibal, (st_p, st_z, st_lat, st_lon) in enumerate(list_streams):
		### Copy P and Z 
		st_p_cut = st_p.copy().split()
		st_z_cut = st_z.copy().split()

		### Ensure time of z is bigger than time of P 
		st_p_cut.trim(starttime = st_z_cut[0].stats.starttime, endtime = st_z_cut[0].stats.endtime, nearest_sample=False)

		### Create stream for new Z data
		st_z_aligned = Stream()
		stats = Stats()
		stats.station = st_z_cut[0].stats.station
		stats.network = st_z_cut[0].stats.network
		stats.channel = st_z_cut[0].stats.channel
		stats.location = st_z_cut[0].stats.location

		### Interpolate between gaps.
		dtz = st_z_cut[0].stats.delta
		dtp = st_p_cut[0].stats.delta 
		Nz = st_z_cut[0].stats.npts

		#### Upsample z 
		znew, tnew = resample(st_z_cut[0].data, t=st_z_cut[0].times(), num=int((dtz/dtp)*Nz +1), window=('tukey', 0.05))

		### Re-interpolate z on same times as P 
		# intalt = interp1d(tnew, znew, kind='cubic')
		intalt = Akima1DInterpolator(tnew, znew)#, method="makima")
		z_aligned = intalt(st_p_cut[0].times())

		### Update stats 
		stats.starttime = st_p_cut[0].stats.starttime
		stats.delta = dtp
		stats.npts = z_aligned.size
			
		### Add data to new Z stream 
		tr_z_aligned = Trace(data=z_aligned, header=stats)
		st_z_aligned.append(tr_z_aligned)

		if do_plot: 
			fig, ax = plt.subplots(figsize=(8,3))
			ax.plot(st_z_aligned[0].times(), st_z_aligned[0].data, c="navy", lw=1, label="Altitude")
			ax.plot(st_z_cut[0].times(), st_z_cut[0].data, c="navy", marker="d", markersize=1, label="Altitude, raw", ls="")
			ax.plot([],[], c="k", lw=1, label="Pressure")
			axb = ax.twinx()
			axb.plot(st_p_cut[0].times(), st_p_cut[0].data, c="k", lw=1)
			ax.set_xlabel("Time / [s]")
			axb.set_ylabel("Pressure / [Pa]")
			ax.set_ylabel("Altitude / [km]")
			ax.legend()

			fig, ax = plt.subplots(figsize=(5,5))
			f_raw = rfftfreq(n=st_z_cut[0].data.size, d=dtz) 
			s_raw = abs(rfft(st_z_cut[0].data))*np.sqrt(2*dtz/st_z_cut[0].data.size)
			###
			f_up = rfftfreq(n=tr_z_aligned.data.size, d=dtp) 
			s_up = abs(rfft(tr_z_aligned.data))*np.sqrt(2*dtp/tr_z_aligned.data.size)
			###
			ax.plot(f_raw, s_raw, c="k", lw=1.5, label="Altitude, raw")
			ax.plot(f_up, s_up, c="navy", lw=1, label="Altitude")
			ax.set_xlabel("Frequency / [Hz]")
			ax.set_ylabel("Altitude spectrum / [km]")
			ax.set_yscale("log")
			ax.set_xscale("log")
			ax.legend()
			plt.show()

		list_streams[ibal][1] = st_z_aligned
		list_streams[ibal][0] = st_p_cut 
		list_streams[ibal].append(st_z_cut)

	return(list_streams)


#############################################################################################################
### STEP FOUR: DTERMINE RELATION BETEEN P AND Z 
def P_Z_relation(list_streams_aligned, do_plot=False, method="piecewise", period_window=500):
	### List to store modeled pressure 
	P_opt = [] 

	for ibal, (st_p, st_z, _, _, _) in enumerate(list_streams_aligned):
		###############################################
		### Option 1: Modify barometric relation to fit 
		### A simple correction of the exponential barometric equation
		def corr_baro(z_data, p_data):
			P0 = 1013.25e2  # Pa (sea level standard pres in mbar.)
			Cp = 1004.68506 # J/K/kg (isobaric heat capacity)
			T0 = 288.16     # K (sea level standard temp.)
			M = 0.02896968  # kg/mol (dry air molar mass)
			R = 8.314462618 # J/(mol.K) (universal gas constant)
			g = 9.80665     # m/s^2 (earth surface gravity)
			def P_z(z, b,a):
				res = P0*(1-g*z/(Cp*T0*b))**(Cp*a*M/R)
				return(res)
			popt, pcov = curve_fit(P_z, z_data, p_data, p0=[1,1], bounds=([0.5,0.5],[2,2]))
			# print("P/z relation optimized for: ",popt)

			p_corr = P_z(z_data, *popt)
			return(p_corr)

		###############################################
		### Option 2: simple linear fit. 
		def corr_linear(z_data, p_data):
			plin = np.polyfit(z_data, p_data, deg=1)
			p_corr = z_data * plin[0] + plin[1] 
			return(p_corr)

		###############################################
		### Option 3: Linear fit, but with a sliding window as large as an oscillation of 100s (half the NBO)
		def corr_window(z_data, p_data, dtp, period_window):
			#period_window = 100
			### Shorter windows lead to more HF noise but better LF correction 

			nwin = int(period_window//dtp)  ### Number of points in window
			Ntr = p_data.size
			p_slide = []
			for i in range(Ntr):
				if i<nwin//2:
					i1 = 0
					i2 = nwin 
				elif i>Ntr-nwin/2:
					i1 = -nwin
					i2 = Ntr
				else:
					i1 = int(i-nwin//2) 
					i2 = int(i+nwin//2) 
				slice_z = z_data[i1:i2]
				slice_p = p_data[i1:i2]
				#print(slice_z.size, i1, i2)

				pslice = np.polyfit(slice_z, slice_p, deg=1)
				p_slide.append(z_data[i] * pslice[0] + pslice[1])
			p_corr = np.array(p_slide)
			return(p_corr)

		###############################################
		P_opt_baro  = st_p.copy()
		P_opt_lin   = st_p.copy()
		P_opt_slide = st_p.copy()
		### Replace pressure data 
		P_opt_baro[0].data  = corr_baro(st_z[0].data, st_p[0].data) 
		P_opt_lin[0].data   = corr_linear(st_z[0].data, st_p[0].data)
		P_opt_slide[0].data = corr_window(st_z[0].data, st_p[0].data, st_p[0].stats.delta, period_window)

		if do_plot:
			fig, ax = plt.subplots(figsize=(6,6))
			ax.scatter(st_z[0].data, st_p[0].data, c="k", lw=1, s=1, alpha=0.1)
			ax.plot(st_z[0].data, st_p[0].data, c="k", lw=1, alpha=0.1)
			ax.plot(st_z[0].data, P_opt_slide[0].data, c="teal", lw=0.8, ls='--', label="Linear, sliding window\n{:d} s".format(period_window))
			ax.plot(st_z[0].data, P_opt_baro[0].data, c="r", lw=2, label="Barometric relation")
			ax.plot(st_z[0].data, P_opt_lin[0].data, c="b", lw=2, ls='--', label="Linear relation")
			ax.set_xlabel("Altitude / [km]")
			ax.set_ylabel("Pressure / [Pa]")
			ax.legend(frameon=False)
		
		if method=="baro":
			list_streams_aligned[ibal].append(P_opt_baro)
		elif method=="linear":
			list_streams_aligned[ibal].append(P_opt_lin)
		elif method=="piecewise":
			list_streams_aligned[ibal].append(P_opt_slide)
	
	return(list_streams_aligned)


#############################################################################################################
### STEP FIVE: OBTAIN CORRECTED PRESSURE 
def P_corrected(list_streams_opt, do_plot=False, method="piecewise"):
	
	for ibal, (st_p, st_z, _, _, _, st_p_opt) in enumerate(list_streams_opt):
		##############################################################################################################################################
		### Substract corrected pressure from original 
		P_corr = st_p.copy()
		P_corr[0].data = st_p[0].data -st_p_opt[0].data
		list_streams_opt[ibal].append(P_corr)

		##############################################################################################################################################
		### Plot it 
		if do_plot:
			fig = plt.figure(figsize= (6,8))
			gs = fig.add_gridspec(4, 1)
			ax1 = fig.add_subplot(gs[0, :])
			ax2 = fig.add_subplot(gs[1, :])
			ax3 = fig.add_subplot(gs[2:, :])
			###
			times_p = st_p[0].times()

			### Pressure signals and predicted pressure
			ax1.plot(times_p, st_p[0].data,  c="k", lw=1, label="Pressure")
			ax1.plot(times_p, st_p_opt[0].data,     c="r", ls="--", label="Pressure predicted")
			ax1.set_ylabel("Pressure / [Pa]")
			ax1.legend(frameon=False)

			### Residuals 
			ax2.plot(times_p, st_p[0].data, c="k", lw=1, label="Pressure")
			ax2.plot([],[], c="r", lw=1, label="Pressure corrected")
			axb = ax2.twinx()
			### Plot the different corrected pressure. 
			axb.plot(times_p, P_corr[0].data, c="r", lw=1)
			ax2.set_xlabel("Time / [s]")
			ax2.set_ylabel("Pressure / [Pa]")
			axb.set_ylabel("Pressure corrected / [Pa]")
			ax2.legend(frameon=False) 

			### Calculate PSD (need to account for gaps)
			win = st_p[0].data.size//8
			dtp = st_p[0].stats.delta
			sp_orig, f_orig =  mlab.psd(st_p[0].data, Fs=1/dtp, NFFT=win, noverlap = win//4,
												detrend='linear', scale_by_freq=True, window=mlab.window_hanning)
			sp_corr, f_corr =  mlab.psd(P_corr[0].data, Fs=1/dtp, NFFT=win, noverlap = win//4,
												detrend='linear', scale_by_freq=True, window=mlab.window_hanning)

			ax3.plot(f_orig, sp_orig, c="k", lw=1, label="Pressure")
			ax3.plot(f_corr, sp_corr, c="r", lw=1, ls="--", label="Pressure corrected")
			ax3.set_xlabel("Frequency / [Hz]")
			ax3.set_xscale("log")
			ax3.set_yscale("log")
			ax3.set_ylabel("PSD [Pa$^2$/Hz]")
			ax3.legend(frameon=False)

			fig.tight_layout()
			# fig.savefig("./FIGURES/Denoising_spectrum_{}.png".format(st_p_cut[0].stats.location), dpi=600)


			#####################################################################
			### Plot FTAN (wavelet spectrogram)
			min_freq = 1e-3 
			max_freq = 0.5

			### Select which type of correction we want to diplay: 
			title = method
			
			fig = plt.figure(figsize= (12,8))
			gs = fig.add_gridspec(2, 2)
			ax1 = fig.add_subplot(gs[0, 0])
			ax2 = fig.add_subplot(gs[0, 1],sharex=ax1)
			ax3 = fig.add_subplot(gs[1, 0])
			ax4 = fig.add_subplot(gs[1, 1])
			cmap ="magma"
			font = 10 

			ax1, _ = plot_scalogram(st_p[0],        ax1, ax_cb = None, font=font, dbrange = [-60, 30])
			ax2, _ = plot_scalogram(P_corr[0],  ax2, ax_cb = None, font=font, dbrange = [-60, 30])
			ax3, _ = plot_scalogram(st_z[0],         ax3, ax_cb = None, font=font, dbrange = ["min", "max"])	
			ax4, _ = plot_scalogram(st_p_opt[0],    ax4, ax_cb = None, font=font, dbrange = [-60, 30])

			axs = [ax1, ax2, ax3, ax4]
			for ax in axs:
				ax.set_ylim(min_freq, max_freq)
				ax.set_ylabel("Frequency / [Hz]")
				#ax1.set_yscale("log")

			ax3.set_xlabel("Time / [s]")
			ax4.set_xlabel("Time / [s]")
			###
			ax1.set_title("Pressure, raw", loc="left")
			ax2.set_title("Pressure, corrected", loc="left")
			ax4.set_title("Pressure, modeled from altitude", loc="left")
			ax3.set_title("Altitude, upsampled", loc="left")

			fig.tight_layout()

	return(list_streams_opt)


#############################################################################################################
### STEP FIVE: OBTAIN CORRECTED PRESSURE 
def save_streams_inversion(destination, list_streams_corr, method="piecewise", period_window=500):

	st_pressure = list_streams_corr[0][6]
	st_alt = list_streams_corr[0][1]
	st_pressure_raw = list_streams_corr[0][0]
	st_lat = list_streams_corr[0][2]
	st_lon = list_streams_corr[0][3]
	st_zgps = list_streams_corr[0][4]
	for ibal, (st_p, st_z, st_l, st_L, st_z2 , _, st_p_corr) in enumerate(list_streams_corr[1:]):
		st_pressure_raw.append(st_p[0])
		st_pressure.append(st_p_corr[0])
		st_alt.append(st_z[0])
		st_lat.append(st_l[0])
		st_lon.append(st_L[0])
		st_zgps.append(st_z2[0])

	### Check that all stream have the same start time (important !)
	tstarts = [] 	
	for i, tr in enumerate(st_pressure):
		tstarts.append(tr.stats.starttime.timestamp)
	for i, tr in enumerate(st_pressure):
		tr.trim(starttime = UTCDateTime(max(tstarts)) )
	for i, tr in enumerate(st_pressure_raw):
		tr.trim(starttime = UTCDateTime(max(tstarts)) )
	for i, tr in enumerate(st_alt):
		tr.trim(starttime = UTCDateTime(max(tstarts)) )
		
	for tr in st_lat:
		if isinstance(tr.data, np.ma.masked_array):
			tr.data = tr.data.filled()
	for tr in st_lon:
		if isinstance(tr.data, np.ma.masked_array):
			tr.data = tr.data.filled()
	for tr in st_zgps:
		if isinstance(tr.data, np.ma.masked_array):
			tr.data = tr.data.filled()
	if method=="piecewise":
		st_pressure.write(destination + "Corrected_pressure_" + method + "_w{:d}.mseed".format(period_window), format="MSEED")
	else:
		st_pressure.write(destination + "Corrected_pressure_" + method + ".mseed", format="MSEED")
	st_pressure_raw.write(destination + "Raw_pressure.mseed", format="MSEED")
	# st_alt.write(destination + "Upsampled_altitude.mseed", format="MSEED")
	st_lat.write(destination + "GPS_latitude.mseed", format="MSEED")
	st_lon.write(destination + "GPS_longitude.mseed", format="MSEED")
	st_zgps.write(destination + "GPS_altitude.mseed", format="MSEED")

	return()

	

### PLOT FOR ARTICLE 
#############################################################################################################
def plot_article_correction():

	fig = plt.figure(figsize=(11,6))
	fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.96, hspace=0.6, wspace=0.4)
	grid = fig.add_gridspec(4, 7, height_ratios=[1,1,1,1], width_ratios=[1,1,0.05,1,1,1,1/10])
	ax_time = fig.add_subplot(grid[0, :2])
	ax_time2 = fig.add_subplot(grid[1, :2])
	ax_freq = fig.add_subplot(grid[2:, :2])
	ax_sp1 = fig.add_subplot(grid[:2, 3:6])
	ax_sp2 = fig.add_subplot(grid[2:, 3:6])
	ax_cb = fig.add_subplot(grid[2:, 6])

	### Load corrected data: 
	st_p = read("./Flores_data/Corrected_pressure_piecewise_w500.mseed")
	st_p_raw = read("./Flores_data/Raw_pressure.mseed")
	st_alt = read("./Flores_data/GPS_altitude.mseed")

	t1 = UTCDateTime("2021-12-14T2:50")
	t2 = UTCDateTime("2021-12-14T4:50")
	st_alt.trim(starttime = t1, endtime=t2)
	st_p.trim(starttime = t1, endtime=t2)
	st_p_raw.trim(starttime = t1, endtime=t2)

	st_predict = st_p_raw.copy() 
	for it, tr in enumerate(st_predict):
		tr.data = st_p_raw[it].data - st_p[it].data 
	print(st_predict)

	### Get balloon names
	balloon_names = [tr.stats.station + " " + tr.stats.location for tr in st_p]
	print(balloon_names)

	times_p = st_p[0].times()

	### Pressure signals and predicted pressure
	ax_time.plot(times_p, st_p_raw[0].data,  c="k", lw=1, label="Raw")
	ax_time.plot(times_p, st_predict[0].data,     c="palevioletred", ls="--", lw=1, label="Predicted")
	ax_time.set_ylabel("Pressure / [Pa]")
	ax_time.legend(frameon=False, loc=3, fontsize=9, ncol=2)
	ax_time2.plot(st_alt[0].times(), st_alt[0].data/1e3,     c="navy", ls="-", label="Altitude")
	ax_time2.set_xlabel("Time / [s from 02:50]")
	ax_time2.set_ylabel("Altitude / [km]")

	ax_time.spines[['right', 'top']].set_visible(False)
	ax_time2.spines[['right', 'top']].set_visible(False)
	

	### Calculate PSD (need to account for gaps)
	win = st_p[0].data.size//8
	dtp = st_p[0].stats.delta
	sp_orig, f_orig =  mlab.psd(st_p_raw[0].data, Fs=1/dtp, NFFT=win, noverlap = win//4,
										detrend='linear', scale_by_freq=True, window=mlab.window_hanning)
	sp_corr, f_corr =  mlab.psd(st_p[0].data, Fs=1/dtp, NFFT=win, noverlap = win//4,
										detrend='linear', scale_by_freq=True, window=mlab.window_hanning)

	ax_freq.plot(f_orig, sp_orig, c="k", lw=1, label="Pressure, raw")
	ax_freq.plot(f_corr, sp_corr, c="crimson", lw=1, ls="--", label="Pressure corrected")
	ax_freq.set_xlabel("Frequency / [Hz]")
	ax_freq.set_xscale("log")
	ax_freq.set_yscale("log")
	ax_freq.set_ylabel("PSD [Pa$^2$/Hz]")
	ax_freq.legend(frameon=False)
	ax_freq.grid(ls=":")
	ax_freq.set_xlim(None, 0.5)

	#########################################
	font = 10 
	from cmcrameri import cm as cmc
	cmap = cmc.lipari

	ax_sp2, _ = plot_scalogram(st_p[0],      ax_sp2, ax_cb = None, font=font, dbrange = [-55, 35], cmap=cmap)
	ax_sp1, ax_cb = plot_scalogram(st_p_raw[0],  ax_sp1, ax_cb = ax_cb, font=font, dbrange = [-55, 35], cmap=cmap)

	axs = [ax_sp1, ax_sp2]
	min_freq = 1e-3 
	max_freq = 0.5
	for ax in axs:
		ax.set_ylim(min_freq, max_freq)
		ax.set_ylabel("Frequency / [Hz]")
		#ax1.set_yscale("log")
	ax_sp1.set_xticklabels([])
	import matplotlib.dates as mdates
	ax_sp2.get_xaxis().set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax_sp2.set_xlabel("Time / [s]")
	###
	ax_sp1.set_title("Pressure, raw", loc="left")
	ax_sp2.set_title("Pressure, corrected", loc="left")
	###
	fig.align_labels()
	fig.savefig("./Figures_article/Strateole2_correction_example.png")
	fig.savefig("./Figures_article/Strateole2_correction_example.pdf")

	return()


#############################################################################################################
def plot_article_picks():
	
	catalog = read_events("./Flores_data/flores_catalog_entry.xml", format="QUAKEML")

	t_event = catalog[0].origins[0].time 
	t_estamp = t_event.timestamp/24/3600
	m_e = catalog[0].magnitudes[0].mag
	print("Magnitude: ", m_e) 
	print("Time: ", t_event.strftime("%d-%m-%y"))

	### Load picks 
	import pickle 
	pdir = "./chains_emcee_floresballoon/DATA/" 
	files_to_open = ["mcmcdata_sigmaS", "mcmcdata_arrivalS", "mcmcdata_sigmaRW", 
						"mcmcdata_arrivalRW", "mcmcdata_sigmaP", "mcmcdata_arrivalP", "mcmcdata_periodRW"]
	data_to_open = ["std_Ss", "arrival_Ss", "std_RWs", "arrival_RWs", "std_Ps", "arrival_Ps", "periods_RWs"]
	class PIK(object):
		def __init__(self):
			for fi, ds in enumerate(files_to_open):
				with open(pdir + ds + "_pik", "rb") as fp:
					dd = pickle.load(fp)
					setattr(self, data_to_open[fi], dd)
	p = PIK()

	### Load corrected data: 
	st_p = read("./Flores_data/Corrected_pressure_piecewise_w500.mseed")
	
	t1 = UTCDateTime("2021-12-14T3:05")
	t2 = UTCDateTime("2021-12-14T3:45")
	# st_alt.trim(starttime = t1, endtime=t2)
	st_p.taper(0.05, type='hann')
	st_p_filt = st_p.copy().filter("bandpass", freqmin=0.06, freqmax=0.2, zerophase=True)
	st_p.trim(starttime = t1, endtime=t2)

	### Get balloon names
	balloon_names = [tr.stats.station + " " + tr.stats.location for tr in st_p]

	from cmcrameri import cm as cmc
	cmap = cmc.lipari
	font = 10 

	#########################################
	for ibal in range(len(st_p)):
		fig = plt.figure(figsize=(7.5,6/1.5))
		fig.subplots_adjust(bottom=0.12, top=0.93, left=0.12, right=0.9, hspace=0.3, wspace=0.1)
		grid = fig.add_gridspec(3, 2, height_ratios=[1,1,1], width_ratios=[1,1/40])
		ax_spectro = fig.add_subplot(grid[:2, 0])
		ax_time = fig.add_subplot(grid[2, 0], sharex=ax_spectro)
		ax_cb = fig.add_subplot(grid[:2, 1])

		ax_spectro, ax_cb = plot_scalogram(st_p[ibal], ax_spectro, ax_cb = ax_cb, font=font, dbrange = [-50, 20], cmap=cmap)
		ax_time.plot(st_p_filt[ibal].times("matplotlib"), st_p_filt[ibal].data, c="k", lw=1)

		### EVENT
		ax_time.axvline(t_estamp, c="k", ls="--")
		ax_spectro.axvline(t_estamp, c="w", ls="--")
		freq_max=0.5
		ax_spectro.text(t_estamp-20/24/3600, 0.95*freq_max, "Mw {:.1f}".format(m_e),
						va="top", ha="right", color="w", fontsize = font,
						bbox=dict(facecolor='white', pad=0, linewidth=0, alpha=0.1))
		
		ax_spectro.text(t_estamp+ (p.arrival_Ps[ibal]-20)/24/3600, 0.95*freq_max, "P",
						va="top", ha="right", color="w", fontsize = font,
						bbox=dict(facecolor='white', pad=0, linewidth=0, alpha=0.1))
		#if ibal !=0 and ibal !=3:
		ax_spectro.text(t_estamp+ (p.arrival_Ss[ibal]-20)/24/3600, 0.95*freq_max, "S",
						va="top", ha="right", color="w", fontsize = font,
						bbox=dict(facecolor='white', pad=0, linewidth=0, alpha=0.1))
		
		if ibal != 2 and ibal !=3:
			ax_spectro.text(t_estamp+ (p.arrival_RWs[ibal][0]+50)/24/3600, 1.5*1/p.periods_RWs[ibal][0], "RW",
							va="bottom", ha="left", color="w", fontsize = font,
							bbox=dict(facecolor='white', pad=0, linewidth=0, alpha=0.1))
		if ibal ==1:
			ax_spectro.text(UTCDateTime("2021-12-14T03:30:00").timestamp/24/3600, 0.95*freq_max, "Gap",
							va="top", ha="center", color="w", fontsize = font,
							bbox=dict(facecolor='white', pad=0, linewidth=0, alpha=0.1))
		
		### PICKS 
		lss = ["--", "-"]
		lws = [1,1.5]
		clss = ["w", "crimson"]
		clps = ["w", "navy"]
		for ls, ax, lw, cls, clp in zip(lss, [ax_spectro, ax_time], lws, clss, clps):
			#if ibal !=0 and ibal !=3:
			ax.axvline(t_estamp + p.arrival_Ss[ibal]/24/3600,color=cls, ls=ls, lw=lw)
			ax.axvline(t_estamp + p.arrival_Ps[ibal]/24/3600,color=clp, ls=ls, lw=lw)
		#if ibal !=0 and ibal !=3:
		ax_time.axvspan(t_estamp + (p.arrival_Ss[ibal]-p.std_Ss[ibal])/24/3600,
						t_estamp + (p.arrival_Ss[ibal]+p.std_Ss[ibal])/24/3600, color="crimson", alpha=0.2)
		ax_time.axvspan(t_estamp + (p.arrival_Ps[ibal]-p.std_Ps[ibal])/24/3600, 
						t_estamp + (p.arrival_Ps[ibal]+p.std_Ps[ibal])/24/3600, color="navy", alpha=0.2)
		
		### Plot extracted curve 
		# ax_spectro.fill_betweenx(1/p.periods_RWs[ibal], 
		# 					t_estamp +(p.arrival_RWs[ibal]-p.std_RWs[ibal])/24/3600, 
		# 					t_estamp +(p.arrival_RWs[ibal]+p.std_RWs[ibal])/24/3600, 
		# 					color="orange", linewidth=0.8, alpha=0.2)
		if ibal != 2 and ibal !=3:
			ax_spectro.errorbar(t_estamp +p.arrival_RWs[ibal]/24/3600, 1/p.periods_RWs[ibal],  
								xerr = p.std_RWs[ibal]/24/3600 , 
								color="w", marker="+", elinewidth=1, capsize=2, lw=1)

		min_freq = 1/200 
		max_freq = 0.5
		ax_spectro.set_ylim(min_freq, max_freq)
		import matplotlib.dates as mdates
		
		ax_time.get_xaxis().set_major_formatter(mdates.DateFormatter('%H:%M'))
		# Rotates and right-aligns the x labels so they don't crowd each other.
		for label in ax_spectro.get_xticklabels(which='major'):
			label.set(rotation=0, horizontalalignment='center')
		ax_time.tick_params(axis='both', which='both', labelsize=font-2)

		ax_time.set_xlabel("Time on 14-12-2021")
		ax_time.set_ylabel("Pressure / [$Pa$]")
		ax_spectro.set_ylabel("Frequency / [Hz]")
		ax_spectro.set_title("Balloon "+ balloon_names[ibal], loc="left")
		ax_cb.set_title("Pressure / [$Pa$]", fontsize=font)
		

		fig.align_labels()
		fig.savefig("./Figures_article/Strateole2_spectrogram_pick_{:d}.png".format(ibal))
		fig.savefig("./Figures_article/Strateole2_spectrogram_pick_{:d}.pdf".format(ibal))


	return()



########################################################################################################
if __name__ == "__main__":
	print("main code")
	# plot_article_correction()
	plot_article_picks()
	plt.show()
