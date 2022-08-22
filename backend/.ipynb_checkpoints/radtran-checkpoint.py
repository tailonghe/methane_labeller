import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import optimize
from scipy import interpolate

# Global variables
c = 299792458.
kB = 1.38064852e-23
deg = np.pi/180
mbartoatm = 0.000986923
kpatoatm = 0.00986923/1000.
kmtocm = 1E5
mtocm = 100.
mol=6.022140857E23
start_time = time.time()

# Key directories
aux_data_dir = '/gdrive/MyDrive/CH4_super_emitter/s2_ch4_code_share/aux_data'
hapi_data_dir = '/gdrive/MyDrive/CH4_super_emitter/s2_ch4_code_share/hapi_data'

def importMRTP(num_layers, targheight, obsheight, solarangle, obsangle, ch4_scale=1.1029, co2_scale=1.2424, h2o_scale=1):
    '''
    Load auxiliary data for vertical profiles of temperature, pressure, and CH4, CO2, and H2O mixing ratio.
    These data come from the U.S. Standard Atmosphere. 
    Also calculate the pathlength in each layer for input observation angles.
    CH4 and CO2 profiles need to be scaled up to match modern-day concentrations.

    Arguments
        num_layers [int]   : Number of vertical pressure layers for the radiative transfer calculation
        targheight [float] : Target elevation above sea level in km
        obsheight  [float] : Altitude of satellite instrument in km. Can use 100 km as if satellite were at top of atmosphere
        solarangle [float] : Solar zenith angle in degrees
        obsangle   [float] : Viewing zenith angle of instrument in degrees
        ch4_scale  [float] : Scale factor multiplying U.S. Standard CH4 profile (1.1029 implies 1875 ppb at the surface)
        co2_scale  [float] : Same but for U.S. Standard CO2 profile (1.2424 implies 410 ppm at the surface)
        h2o_scale  [float] : Same but for U.S. Standard H2O profile

    Returns
        pathlength [float] : Path length in each layer [cm]
        pressavg   [float] : Average pressure in each layer [atm]
        tempavg    [float] : Average temperature in each layer [K]
        mrCH4avg   [float] : Average CH4 mixing ratio in each layer
        mrCO2avg   [float] : Average CO2 mixing ratio in each layer
        mrH2Oavg   [float] : Average H2O mixing ratio in each layer
    ''' 
        
    # Mixing Ratio
    mrCH4 = np.transpose(np.genfromtxt(f'{aux_data_dir}/ch4.dat'))
    mrCO2 = np.transpose(np.genfromtxt(f'{aux_data_dir}/co2.dat'))
    mrH2O = np.transpose(np.genfromtxt(f'{aux_data_dir}/h2o.dat'))
    
    # Temperature
    temp = np.transpose(np.genfromtxt(f'{aux_data_dir}/temperature.dat'))
    
    # Pressure
    press = np.transpose(np.genfromtxt(f'{aux_data_dir}/pressure.dat'))


        # Mixing Ratio
    # mrCH4 = np.transpose(ch4_data)
    # mrCO2 = np.transpose(co2_data)
    # mrH2O = np.transpose(h2o_data)
    
    # Temperature
    # temp = np.transpose(temp_data)
    
    # Pressure
    # press = np.transpose(press_data)
    
    
    # Define altitude in cm
    altitude = press[1][::-1] * kmtocm
    # Define pressure in atm
    pressure = press[0][::-1] * mbartoatm
    
    # Find pressure as a function of altitude
    altfine = np.arange(0, 10, 0.01)
    pressfine = np.interp(altfine, press[1], press[0]*mbartoatm)
    idfine = (np.abs(altfine-targheight)).argmin()
    pressmax = pressfine[idfine]    
    
    # Interpolate to get layers evenly spaced in pressure
    dpress = (pressmax-2.9608E-5)/num_layers
    pressinterp = np.arange(0, pressmax+dpress, dpress)
    altinterp = np.interp(pressinterp, pressure, altitude)

    # Find temperature at each altitude
    alttemp = temp[1]*kmtocm
    temperature = temp[0]
    tempinterp = np.interp(altinterp, alttemp, temperature)
    
    # Find mixing ratios at each altitude
    altmrCH4 = mrCH4[1]*kmtocm
    altmrCO2 = mrCO2[1]*kmtocm
    altmrH2O = mrH2O[1]*kmtocm
    mixrateCH4 = mrCH4[0]*ch4_scale
    mixrateCO2 = mrCO2[0]*co2_scale
    mixrateH2O = mrH2O[0]*h2o_scale
    
    # Interpolate and then sample using the altitude sample points determined from isobaric pressure increases
    mrCH4interp = np.interp(altinterp, altmrCH4, mixrateCH4)
    mrCO2interp = np.interp(altinterp, altmrCO2, mixrateCO2)
    mrH2Ointerp = np.interp(altinterp, altmrH2O, mixrateH2O)
    
    def find_nearest_alt(array,value):
        idx = (np.abs(array-value)).argmin()
        secondpass = array[idx:len(array)]
        zeroarray = np.zeros(idx)
        upwellingpass = np.concatenate((zeroarray, secondpass))
        return upwellingpass
    
    upwellingpass = find_nearest_alt(altinterp, obsheight*kmtocm)
    
    # Find path length of each layer
    pathlengthdown = np.zeros(num_layers)
    pathlengthup = np.zeros(num_layers)
    for i in range(0, num_layers):
        pathlengthdown[i] = np.absolute(altinterp[i]-altinterp[i+1])
        pathlengthup[i] = np.absolute(upwellingpass[i]-upwellingpass[i+1])
        
    # Calculate path given the Solar and observation angle from Nadir
    pathlength = pathlengthdown/np.cos(solarangle*deg) + pathlengthup/np.cos(obsangle*deg)
 
    # Define average value in layers
    pressavg = np.zeros(len(pathlength))
    tempavg = np.zeros(len(pathlength))
    mrCH4avg = np.zeros(len(pathlength))
    mrCO2avg = np.zeros(len(pathlength))
    mrH2Oavg = np.zeros(len(pathlength))   
    for i in range(0,len(pathlength)):
        pressavg[i] = (pressinterp[i+1]+pressinterp[i])/2.
        tempavg[i] = (tempinterp[i+1]+tempinterp[i])/2.
        mrCH4avg[i] = (mrCH4interp[i+1]+mrCH4interp[i])/2.
        mrCO2avg[i] = (mrCO2interp[i+1]+mrCO2interp[i])/2.
        mrH2Oavg[i] = (mrH2Ointerp[i+1]+mrH2Ointerp[i])/2.

    return pathlength, pressavg, tempavg, mrCH4avg, mrCO2avg, mrH2Oavg

def radtran(targheight, obsheight, solarangle, obsangle, instrument, band, num_layers=100):
    '''
    Computes the top-of-atmosphere spectral radiance (TOASR) for an input instrument and spectral band.

    Arguments
        targheight [float] : Target elevation above sea level in km
        obsheight  [float] : Altitude of satellite instrument in km. Can use 100 km as if satellite were at top of atmosphere
        solarangle [float] : Solar zenith angle in degrees
        obsangle   [float] : Viewing zenith angle of instrument in degrees
        instrument [str]   : MSI instrument. Choose 'S2A' or 'S2B'
        band       [int]   : Spectral band. Choose 11 or 12
        num_layers [int]   : Number of vertical pressure layers for the radiative transfer calculation

    Returns
        toasr          [float] : Band-integrated top-of-atmosphere spectral radiance [W/m2/m/sr]
        odCH4pts       [float] : CH4 optical depth by wavelength
        odCO2pts       [float] : CO2 optical depth by wavelength
        odH2Opts       [float] : H2O optical depth by wavelength
        solar_spectrum [float] : Upwelling solar spectrum
        cdCH4          [float] : CH4 slant column density in mol/m2
    '''

    start_time = time.time()   
    
    print('Creating the transmission spectrum...')
    
    # Import pressure, temperature, path-length, and mixing ratios
    (L_cm, press_atm, temp, mrCH4, mrCO2, mrH2O) = importMRTP(num_layers, targheight, obsheight, solarangle, obsangle)
    
    # Load absorption cross_sections        
    wavelength = np.load(f'{hapi_data_dir}/abs_wave_hapi_{instrument}_band{band}.npy')
    press_load = np.load(f'{hapi_data_dir}/abs_press_hapi_{instrument}_band{band}.npy')
    temp_load = np.load(f'{hapi_data_dir}/abs_temp_hapi_{instrument}_band{band}.npy')   
    absCH4_load  = np.load(f'{hapi_data_dir}/abs_ch4_hapi_{instrument}_band{band}.npy')
    absCO2_load = np.load(f'{hapi_data_dir}/abs_co2_hapi_{instrument}_band{band}.npy')
    absH2O_load = np.load(f'{hapi_data_dir}/abs_h2o_hapi_{instrument}_band{band}.npy')
    num_wave = len(wavelength)

    # Get solar spectrum
    solarspec = np.transpose(np.genfromtxt(f'{aux_data_dir}/SUNp01_4000_to_7000.txt'))
    wavesolar = 1E7/solarspec[0][::-1]
    radiancesolar = solarspec[1][::-1]*(100*solarspec[0][::-1]**2)
    solarradiance = np.interp(wavelength,wavesolar,radiancesolar)
    
    # Calculate optical density
    odCH4pts_upper = np.zeros(num_wave)
    odCH4pts_lower = np.zeros(num_wave)
    odCO2pts = np.zeros(num_wave)
    odH2Opts = np.zeros(num_wave)
    
    interp_order = 3
    for i in range(num_wave):
        
        fCH4_tp = scipy.interpolate.RectBivariateSpline(temp_load, press_load, absCH4_load.T[i], kx=interp_order, ky=interp_order)
        fCO2_tp = scipy.interpolate.RectBivariateSpline(temp_load, press_load, absCO2_load.T[i], kx=interp_order, ky=interp_order)
        fH2O_tp = scipy.interpolate.RectBivariateSpline(temp_load, press_load, absH2O_load.T[i], kx=interp_order, ky=interp_order)
               
        for j in range(num_layers):
            
            # Calculate density
            temperature_K = temp[j]
            pressure_atm = press_atm[j]
            pressure_Pa = pressure_atm*101325
            density_m3 = pressure_Pa/(kB*temperature_K)
            density_cm3 = density_m3/(1E6)
            
            # Evaluate interpolation function
            f_CH4_temp = fCH4_tp(temperature_K, pressure_atm)
            f_CO2_temp = fCO2_tp(temperature_K, pressure_atm)
            f_H2O_temp = fH2O_tp(temperature_K, pressure_atm)
                        
            # Calculate the (unit-less) optical density: OD = abs*n*MR*L
            lim_low = 6
            if j >= num_layers - lim_low:
                # Lowest 6 pressure layers = lowest 500 m of atmosphere (lower)
                odCH4pts_lower[i] = odCH4pts_lower[i] + f_CH4_temp*density_cm3*mrCH4[j]*L_cm[j]
            else:
                # The rest of the atmosphere (upper)
                odCH4pts_upper[i] = odCH4pts_upper[i] + f_CH4_temp*density_cm3*mrCH4[j]*L_cm[j]
            odCO2pts[i] = odCO2pts[i] + f_CO2_temp*density_cm3*mrCO2[j]*L_cm[j]
            odH2Opts[i] = odH2Opts[i] + f_H2O_temp*density_cm3*mrH2O[j]*L_cm[j]

    # Calculate slant column density of methane
    cdCH4 = np.sum((press_atm/kpatoatm/(kB*temp)/mtocm**3)*mrCH4*L_cm/mol*mtocm**2)

    press_atm_lower = press_atm[-lim_low:]
    temp_lower = temp[-lim_low:]
    L_cm_lower = L_cm[-lim_low:]
    cdCH4_lower = np.sum((press_atm_lower/kpatoatm/(kB*temp_lower)/mtocm**3)*mrCH4[-lim_low:]*L_cm_lower/mol*mtocm**2)

    # Calculate the Top-Of-Atmosphere Spectral Radiance (TOASR) in the band [W/m2/m/sr]
    solar_spectrum = solarradiance/np.pi * np.cos(solarangle*deg)
    toasr = np.mean(np.exp(-(odCH4pts_lower + odCH4pts_upper + odCO2pts + odH2Opts)) * solar_spectrum)
    
    print("--- %s seconds --- to run radtran()" % (time.time() - start_time))

    return toasr, odCH4pts_lower, odCH4pts_upper, odCO2pts, odH2Opts, solar_spectrum, cdCH4, cdCH4_lower

def retrieve(frac_refl_data, instrument, method, targheight, obsheight, solarangle, obsangle, num_layers=100):
    '''
    Infer methane column enhancements from fractional reflectance measurements.

    Arguments
        frac_refl_data [float] : Array of fractional reflectance data, deltaR = (cR-R0)/R0
                                 e.g., DeltaR_SBMP from eq. (1) in Varon et al. 2021 AMT
        instrument     [str]   : MSI instrument. Choose 'S2A' or 'S2B'
        method         [str]   : Retrieval method corresponding to frac_refl_data
                                 Choose 'MBSP' or 'SBMP'
        targheight     [float] : Target elevation above sea level in km
        obsheight      [float] : Altitude of satellite instrument in km. Can use 100 km as if satellite were at top of atmosphere
        solarangle     [float] : Solar zenith angle in degrees
        obsangle       [float] : Viewing zenith angle of instrument in degrees
        num_layers     [int]   : Number of vertical pressure layers for the radiative transfer calculation
    '''

    # AJT: Choose bands depending on instrument
    if instrument == 'S2A' or instrument == 'S2B':
        bandA = 11
        bandB = 12
    elif instrument == 'L4' or instrument == 'L5' or instrument == 'L7':
        bandA = 5
        bandB = 7
    elif instrument == 'L8' or instrument == 'L9':
        bandA = 6
        bandB = 7
    else:
        raise ValueError('Bad instrument selection. Must be "S2A", "S2B", "L4", "L7", or "L8".')
    # Choose method
    print('RB: ',bandA)

    if method == 'SBMP':
        
        # Get toasr, optical depths, etc. from radtran()
        toasr_12, odCH4_lower_12, odCH4_upper_12, odCO2_12, odH2O_12, solar_spectrum_12, cdCH4, cdCH4_lower = radtran(targheight, obsheight, solarangle, obsangle, instrument, band=bandB, num_layers=100)

        def frac_abs_SBMP_difference(ch4_enh, data): 
            '''
            Fractional absorption model to compare with measurements for SBMP method.
            
            Arguments
                ch4_enh [float] : Modeled enhancement as fraction of background
                data    [float] : Actual (cR-R0)/R0
            '''
                
            ch4 = ch4_enh + 1
            toasr_CH4enh_12 = np.mean(np.exp(-(ch4*odCH4_lower_12 + odCH4_upper_12 + odCO2_12 + odH2O_12))*solar_spectrum_12)
            frac_abs_SBMP = (toasr_CH4enh_12 - toasr_12)/toasr_12
            
            return frac_abs_SBMP - data
    
    elif method == 'MBSP':

        # Get toasr, optical depths, etc. from radtran()
        toasr_11, odCH4_lower_11, odCH4_upper_11, odCO2_11, odH2O_11, solar_spectrum_11, _, _ = radtran(targheight, obsheight, solarangle, obsangle, instrument, band=bandA, num_layers=num_layers)
        toasr_12, odCH4_lower_12, odCH4_upper_12, odCO2_12, odH2O_12, solar_spectrum_12, cdCH4, cdCH4_lower = radtran(targheight, obsheight, solarangle, obsangle, instrument, band=bandB, num_layers=num_layers)
        
        def frac_abs_MBSP_difference(ch4_enh, data):
            '''
            Fractional absorption model to compare with measurements for MBSP method.
            
            Arguments
                ch4_enh [float] : Modeled enhancement as fraction of background
                data    [float] : Actual (cR-R0)/R0
            '''
                
            ch4 = ch4_enh + 1
            toasr_CH4enh_12 = np.mean(np.exp(-(ch4*odCH4_lower_12 + odCH4_upper_12 + odCO2_12 + odH2O_12))*solar_spectrum_12)
            toasr_CH4enh_11 = np.mean(np.exp(-(ch4*odCH4_lower_11 + odCH4_upper_11 + odCO2_11 + odH2O_11))*solar_spectrum_11)

            frac_abs_12 = (toasr_CH4enh_12 - toasr_12)/toasr_12
            frac_abs_11 = (toasr_CH4enh_11 - toasr_11)/toasr_11
            
            frac_abs_MBSP = frac_abs_12 - frac_abs_11 
                
            return frac_abs_MBSP - data

    else:
        raise ValueError('Bad method selection. Must be "MBSP" or "SBMP".')
     
    start_time = time.time()

    # Do retrieval
    (num_rows, num_cols) = frac_refl_data.shape
    ch4_out = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):

            data_temp = frac_refl_data[i,j]
                
            if np.isnan(data_temp):
                print('Found nan, skipping.')
                ch4_out[i,j] = np.nan
            
            # Solve for the best-fit fractional column (fraction of background)
            else:
                if method == 'SBMP':
                    ch4_temp = scipy.optimize.newton(lambda ch4_scale: frac_abs_SBMP_difference(ch4_scale,data_temp), 0, rtol=0.0001, maxiter=10000, disp=False)
                elif method == 'MBSP':
                    ch4_temp = scipy.optimize.newton(lambda ch4_scale: frac_abs_MBSP_difference(ch4_scale,data_temp), 0, rtol=0.0001, maxiter=10000, disp=False)

                # Convert the fractional column to absolute vertical column density in mol/m2   
                AMF = 1/np.cos(obsangle*deg) + 1/np.cos(solarangle*deg)
                ch4_out[i,j] = ch4_temp * (cdCH4_lower/cdCH4)*(cdCH4/AMF)
        
    # Time    
    print("--- %s seconds --- to optimize" % (time.time() - start_time))

    return ch4_out