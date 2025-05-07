import MCMC_modules as mc
import numpy as np
import os
from obspy import read, Stream
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.core.event import read_events


#############################################################################################################
def get_flores_event(method="piecewise", period_window=500, do_plot=False):

    ### Find the intercepts of two curves, given by the same x data
    def interpolated_intercepts(x, y1, y2, **kwargs):

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        def intercept(point1, point2, point3, point4):
            """find the intersection between two lines
            the first line is defined by the line between point1 and point2
            the second line is defined by the line between point3 and point4
            each point is an (x,y) tuple.

            So, for example, you can find the intersection between
            intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

            Returns: the intercept, in (x,y) format
            """    

            L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
            L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

            R = intersection(L1, L2)

            return R

        idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

        xcs = []
        ycs = []
        if 'aux' in kwargs : 
            auxs = []

        for idx in idxs:
            xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
            xcs.append(xc)
            ycs.append(yc)
            if 'aux' in kwargs:
                ds = []
                for data in kwargs['aux']:
                    y1, y2 = data[idx], data[idx+1]
                    x1, x2 = x[idx], x[idx+1] 
                    d = (y2-y1)/(x2-x1) * xc - (y2*x1 - x2*y1)/(x2-x1)
                    ds.append(d)
                auxs.append(ds)
        if 'aux' in kwargs:
            return np.array(xcs), np.array(ycs), np.array(auxs)
        else:
            return np.array(xcs), np.array(ycs)
        

    ### Code to find the time at which a wave and a balloon, both travelling, intercept each other. 
    def traj_balloon_wave(tr_time, tr_lat, tr_lon, tr_alt, v_wave, t_event, lat_event, lon_event, h_event) : 
        ### Approx. Speed in air
        v_air = 340

        ### Distance balloon to source at t=t_event
        it_1 = np.where(abs(tr_time - t_event.timestamp)== np.min(np.abs(tr_time - t_event.timestamp)))[0][0]
        d0_balloon, *_ = gps2dist_azimuth(lat_event, lon_event, tr_lat[it_1], tr_lon[it_1])

        ### Boundary time: The RW has time to travel twice to the balloon 
        it_2 = np.where(abs(tr_time - t_event.timestamp - 2*d0_balloon/v_wave)== \
                                np.min(np.abs(tr_time - t_event.timestamp - 2*d0_balloon/v_wave)))[0][0]
        
        dist = [d0_balloon]
        height = tr_alt[it_1:it_2+1]
        time = tr_time[it_1:it_2+1]-t_event.timestamp
        for i in range(it_1+1,it_2+1):
            dist.append(gps2dist_azimuth(lat_event, lon_event, tr_lat[i], tr_lon[i])[0])  
        dist = np.array(dist)

        t_rayleigh_to_balloon = h_event/v_wave + dist/v_wave + height/v_air 

        intersect = np.where(abs(t_rayleigh_to_balloon-time) == \
                                np.min(abs(t_rayleigh_to_balloon-time)))[0][0]

        ### With finer resolution, between data points 
        xcs, ycs, auxs = interpolated_intercepts(time, time, t_rayleigh_to_balloon, 
                        aux = [tr_lat[it_1:it_2+1],tr_lon[it_1:it_2+1],tr_alt[it_1:it_2+1] ])

        t_intercept = xcs[0][0] + t_event.timestamp 
        d_intercept = gps2dist_azimuth(lat_event, lon_event, auxs[:,0], auxs[:,1])[0]
        lat_intercept = auxs[:,0][0][0]
        lon_intercept = auxs[:,1][0][0]
        h_intercept = auxs[:,2][0][0]
        # print(t_intercept, d_intercept, h_intercept)

        # fig, ax = plt.subplots() 
        # ax.plot(time, time, 'k')
        # ax.plot(time, t_rayleigh_to_balloon, 'r', marker='o')
        # ax.plot(time[intersect], t_rayleigh_to_balloon[intersect], 'k', marker='s', ls='')
        # ax.plot(xcs, ycs, 'b', marker='s', ls='')
        # ax.plot(t_intercept-t_event.timestamp, h_event/v_wave + d_intercept/v_wave + h_intercept/v_air, 'g', marker='^', ls='')
        # fig, ax = plt.subplots() 
        # ax.plot(time, tr_alt[it_1:it_2+1], 'r', marker='o')
        # ax.plot(xcs, auxs[:,2], 'b', marker='s', ls='')
        # plt.show()

        return(t_intercept, lat_intercept, lon_intercept, h_intercept)


    ### Load event data 
    catalog = read_events("./Flores_data/flores_catalog_entry.xml", format="QUAKEML")
    lat_event = catalog[0].origins[0].latitude
    lon_event = catalog[0].origins[0].longitude
    dep_event = catalog[0].origins[0].depth
    t_event = catalog[0].origins[0].time 
    m_e = catalog[0].magnitudes[0].mag
    print("Magnitude: ", m_e) 
    print("Time: ", t_event.strftime("%d-%m-%y"))

    ### Load corrected balloon data: 
    if method == "piecewise":
        st_p = read("./Flores_data/Corrected_pressure_" + method + "_w{:d}.mseed".format(period_window))
    else:
        st_p = read("./Flores_data/Corrected_pressure_" + method + ".mseed")
    st_p_raw = read("./Flores_data/Raw_pressure.mseed")
    st_lat = read("./Flores_data/GPS_latitude.mseed")
    st_lon = read("./Flores_data/GPS_longitude.mseed")
    st_alt = read("./Flores_data/GPS_altitude.mseed")

    ### Get balloon names
    balloon_names = [tr.stats.station + " " + tr.stats.location for tr in st_p]
    print("Balloons: ", balloon_names)

    ### Find balloon (station) location at approximate arrival. 
    balloon_lats = []   
    balloon_lons = []   
    balloon_alts = []
    balloon_dist = []
    stream_save = Stream()

    for ibal in range(len(balloon_names)):
        tr_time = st_lat[ibal].times("timestamp")
        tr_lat = st_lat[ibal].data 
        tr_lon = st_lon[ibal].data 
        tr_alt = st_alt[ibal].data
        ### Calculate travel time for P and RW (approximate)
        v_rw = 4000
        v_p = 8e3
        ###
        t_intercept_rw, lat_rw, lon_rw, alt_rw  = traj_balloon_wave(tr_time, tr_lat, tr_lon, tr_alt, v_rw, t_event, lat_event, lon_event, dep_event) 
        t_intercept_p, lat_p, lon_p, alt_p      = traj_balloon_wave(tr_time, tr_lat, tr_lon, tr_alt, v_p, t_event, lat_event, lon_event, dep_event) 
    
        print("Approximate predicted time difference between P and RW: ", t_intercept_rw-t_intercept_p)  # 80s 
        print("Approximate predicted distance moved between P and RW: ", gps2dist_azimuth(lat_rw, lon_rw, lat_p, lon_p)[0] ) # 400 m
        print("Approx horizontal speed of balloon: ", gps2dist_azimuth(lat_rw, lon_rw, lat_p, lon_p)[0]/abs(t_intercept_rw-t_intercept_p) )

        balloon_lats.append((lat_rw+lat_p)/2)              
        balloon_lons.append((lon_rw+lon_p)/2)
        balloon_alts.append((alt_rw+alt_p)/2)
        balloon_dist.append( gps2dist_azimuth(lat_event, lon_event, balloon_lats[-1], balloon_lons[-1])[0] /1e3)
        stream_save.append(st_p[ibal])

    # print(stream_save)
    balloon_degs = np.array([kilometers2degrees(s) for s in balloon_dist])

    ### Load Median of Crust1.0 and LLNL models for Flores
    median_crust1 = np.load("./Flores_data/Median_crust1_model_Flores.npy")
    median_llnl = np.load("./Flores_data/Median_llnl_model_Flores.npy") 
    ### Use a composite of the two
    u = np.where( np.cumsum(median_llnl[0,:])> np.sum(median_crust1[0,:])  )[0][0]
    llnl_new = median_llnl.T[u:,:].copy()
    llnl_new[0,0] = np.sum(median_llnl[0,:u+1]) - np.sum(median_crust1[0,:])
    velocity_model_median = np.vstack((median_crust1.T, llnl_new))

    ### Calculate theoretical group velocities for Rayleigh Waves 
    periods = 1/10**np.linspace(-3, 0, 100)[::-1]
    RW_periods, RW_vg = mc.compute_vg_n_layers(periods, velocity_model_median, max_mode = 1,)
    RW_freqs = 1/RW_periods[0]
    RW_vg = RW_vg[0]

    ### Formatting for passing to McMC 
    event_info = [lon_event, lat_event, dep_event, t_event.timestamp]
    stations_info = [balloon_lons, balloon_lats, balloon_alts, balloon_dist, balloon_degs]
    model_info = [velocity_model_median.T, RW_freqs, RW_vg]
    title = "Flores event Mw {:.1f} on ".format(m_e) + t_event.strftime("%d-%m-%y %H:%M")
    
    return  stream_save, event_info, stations_info, model_info, title



#############################################################################################################
def generate_data(data_dir, initialize=False):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ### Get the Flores waveform data from external code 
    if not os.path.isfile(data_dir + "data_flores_event.npy"):
        
        obspy_traces, event_info, stations_info, model_info, tit = get_flores_event()

        ### Saving to text and npy files to gain time 
        np.save(data_dir + "data_flores_event", event_info)
        np.save(data_dir + "data_flores_stations", stations_info)
        np.save(data_dir + "data_flores_model", model_info[0])
        np.save(data_dir + "data_flores_modelRW", model_info[1:])
        obspy_traces.write(data_dir + "data_flores_traces.mseed", format="MSEED")

        np.savetxt(data_dir + "data_flores_event.txt", np.transpose(event_info), fmt='% .3f', header=tit)
        np.savetxt(data_dir + "data_flores_stations.txt", np.transpose(stations_info), fmt='% 5.3f')
        np.savetxt(data_dir + "data_flores_model.txt", np.transpose(model_info[0]), fmt='% 4.3f', header=tit)

    ### Load event and station information 
    event_info                              = np.load(data_dir + "data_flores_event.npy")
    stations_info                           = np.load(data_dir + "data_flores_stations.npy")
    velocity_model_median                   = np.load(data_dir + "data_flores_model.npy")
    RW_freqs, RW_vg                         = np.load(data_dir + "data_flores_modelRW.npy")
    obspy_traces                            = read(data_dir + "data_flores_traces.mseed")

    ### If we want to select only some of the balloons: 
    # select = [2]  
    ### All balloons: 
    select = [i for i in range(len(obspy_traces))]

    ### Select only some traces: 
    obspy_old = obspy_traces.copy() 
    stations_info_old = stations_info 
    obspy_traces = Stream()
    stations_info = []
    for s in select: 
        obspy_traces.append(obspy_old[s])
    for si in stations_info_old: 
        dat = []
        for s in select: 
            dat.append(si[s])
        stations_info.append(dat)

    seismic_info = [obspy_traces, stations_info, event_info, velocity_model_median.T, select ]
    ### NOTE: Traces are supposed to be aligned in time 

    ### Reference time from which the arrival times will be picked: 
    ### If the event time is known, then t_ref=T_event for simplicity. 
    t_ref = event_info[3]
    ### Max, min period for extraction of Rayleigh Wave arrivals 
    Tmin, Tmax = 5, 500 #300
    ### Max, min for McMC 
    Tmin_mcmc, Tmax_mcmc = 5, 1000
    model_atm = np.load("./Atm_models/flores_model.npy")


    #####################################################################################
    ### Prepare the data 
    opt_data = dict(
        ### Reference time to pick arrival time from: 
        t_ref = t_ref, 
        ### Periods for dispersion curve searched between 100s and 5s. 
        target_periods = np.logspace(np.log10(Tmin_mcmc),np.log10(Tmax_mcmc),30),   
        ### Periods for FTAN search 
        periods = np.logspace(np.log10(Tmin),np.log10(Tmax),100), 
        ### If we want to plot the phase picking 
        plot=False,#initialize,
        ### Where to save data extraction results 
        data_dir=data_dir,
        ### The atmospheric model 
        model_atm = model_atm
        )


    ### Pick the S, RW by hand and saves it somewhere if initialize=True
    DATA = mc.MCMC_data(*seismic_info, **opt_data, initialize=initialize)

    ### Stations : ['TTL3 17', 'TTL5 16', 'TTL4 15', 'TTL4 07']
    ### Flores: Empty DATA vector for variables that are not well constrained
    ### TTL3 17 : P, S, RW
    ### TTL5 16 : P, S, RW 
    ### TTL4 15 : P, S, no RW 
    ### TTL4 07 : P, S, barely RW 
    DATA.data_vector[2]["RW_arrival"] = None 
    DATA.data_vector[3]["RW_arrival"] = None  

    return(DATA)



########################################################################################################
### Plot a map of the stations  
if __name__ == "__main__" :

    sn = os.path.basename(__file__)
    fname = sn.split("setup_prior_data_")[1].split(".")[0]
    
    DATA  = generate_data('chains_emcee_' + fname + '/DATA/', initialize=False)#True)
    
    
    def plot_map_stations_event():
        ### To check the azimuth configuration of event/stations 
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt 

        fig, ax = plt.subplots(figsize = (8,8))
        
        ### Make map around stations and events 
        lats_all = DATA.sta_lats + [DATA.ev_lat] 
        lons_all = DATA.sta_lons + [DATA.ev_lon] 
        m = Basemap(ax=ax, projection='lcc', resolution='l', 
                        lat_0=(max(lats_all)+min(lats_all))/2, lon_0=(max(lons_all)+min(lons_all))/2,
                        llcrnrlon= min(lons_all)-5,llcrnrlat=min(lats_all)-5,
                        urcrnrlon=max(lons_all)+10,urcrnrlat=max(lats_all)+5 )
        elat, elon = 5, 10

        ### Draw coastlines and country borders
        m.drawcoastlines(color="#94568c", linewidth =0.5)
        m.drawmapboundary(fill_color='lavender')
        m.fillcontinents(color='darksalmon',lake_color='lavender')

        ### Draw parallels and meridians
        m.drawparallels(range(-20, 10, elat), labels=[1,0,0,0] ,linewidth=0.6, color="grey")
        m.drawmeridians(range(90, 130, elon), labels=[0,0,0,1] ,linewidth=0.6, color="grey")

        ### Convert latitude and longitude to x and y coordinates and plot event 
        x_B, y_B = m(DATA.ev_lon, DATA.ev_lat)
        m.plot(x_B, y_B, markerfacecolor = 'gold', markeredgecolor="k", marker="*", markersize=10)

        ### Plot the stations
        for i , (slat, slon) in enumerate(zip(DATA.sta_lats, DATA.sta_lons)):
            x_A, y_A = m(slon, slat)
            m.plot(x_A, y_A, 'r', marker="^", markeredgecolor="k", markersize=10)
            ### Annotate the points
            ax.text(x_A, y_A, '  {:d}'.format(i), fontsize=12)

        ### Title and show plot
        ax.set_title('Station - event configuration')
        ax.set_ylabel(r"Latitude / [$^{\circ}]$", labelpad=30)
        ax.set_xlabel(r"Longitude / [$^{\circ}]$", labelpad=15)
        
        plt.show()

    plot_map_stations_event()
    
