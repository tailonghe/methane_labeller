import pandas as pd
from datetime import datetime, timedelta
import geemap
import ee
import numpy as np
from sklearn.linear_model import HuberRegressor
import os

from backend.radtran import retrieve


# parameters for radtran
num_layers = 100
targheight = 0
obsheight = 100
solarangle = 40
obsangle = 0
method = 'MBSP'

satellite_database = {
    'Landsat 8': {
      'Folder': 'LC08',
      'Red': 'B4',
      'Green': 'B3',
      'Blue': 'B2',
      'NIR': 'B5',
      'SWIR1': 'B6',
      'SWIR2': 'B7',
      'Cloud': 'B11',
      'Shortname': 'L8',
    },
    'Landsat 7': {
      'Folder': 'LE07',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6_VCID_2',
      'Shortname': 'L7',
    },
    'Landsat 5': {
      'Folder': 'LT05',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6',
      'Shortname': 'L5',
    },
    'Landsat 4': {
      'Folder': 'LT04',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6',
      'Shortname': 'L4',
    },
    'Sentinel-2': {
      'Folder': 'NA',
      'Red': 'B4',
      'Green': 'B3',
      'Blue': 'B2',
      'NIR': 'B8',
      'SWIR1': 'B11',
      'SWIR2': 'B12',
      'Cloud': 'QA60',
      'Shortname': 'S2',
    },
}


def get_s2_cld_col(aoi, start_date, end_date):
    ## Import and filter S2.
    # s2_sr_col = (ee.ImageCollection('COPERNICUS/S2')
    #     .filterMetadata('MGRS_TILE', 'equals', '31SGR')
    #     .filterBounds(aoi)
    #     .filterDate(start_date, end_date)
    #     )

    # Import and filter S2.
    s2_sr_col_all = (ee.ImageCollection('COPERNICUS/S2')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        )
    mgsr_list = s2_sr_col_all.aggregate_array('MGRS_TILE').getInfo()
    print('mgsr_list: ', mgsr_list)
    if len(mgsr_list) == 0:
        return None
    s2_sr_col = s2_sr_col_all.filterMetadata('MGRS_TILE', 'equals', mgsr_list[0])
    s2_sr_img_size = s2_sr_col.size().getInfo()
    print(mgsr_list[0] + ' size: ', s2_sr_img_size)

    for tile in mgsr_list[1:]:
        s2_sr_col_tmp = s2_sr_col_all.filterMetadata('MGRS_TILE', 'equals', tile)
        current_size = s2_sr_col_tmp.size().getInfo()
        print(tile + ' size: ', current_size)
        if current_size > s2_sr_img_size:
            s2_sr_img_size = current_size
            s2_sr_col = s2_sr_col_all.filterMetadata('MGRS_TILE', 'equals', tile)
    
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability').rename('cloud_prob')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb]))

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

def get_plume(tkframe, lon, lat, startDate, endDate, dX=1.5, dY=1.5, do_retrieval=False, satellite='L8'):
    '''
    dX/dY: distance (km) in NS/WE direction
    lon: longitude (~180 -- 180)
    lat: latitude
    startDate/endDate: string ('YYYY-MM-DD') for initial/final date
    do_retrieval: flag for calculating XCH4 using the MBSP approach
    satellite: Satellite name (L4, L5, L7, L8, and S2) 
    '''
    # Initialize Earth Engine
    ee.Initialize()

    # Coordinate mapping for rectangle of plume
    grid_pt = (lat, lon)
    dlat = dY/110.574
    dlon = dX/111.320/np.cos(lat*0.0174532925)
    print('dlat, dlon: ', dlat, dlon)
    W=grid_pt[1]-dlon 
    E=grid_pt[1]+dlon
    N=grid_pt[0]+dlat
    S=grid_pt[0]-dlat
    re   = ee.Geometry.Point(lon, lat)
    region = ee.Geometry.Polygon(
        [[W, N],\
        [W, S],\
        [E, S],\
        [E, N]])

    era5_region = ee.Geometry.Polygon(
            [[lon-0.01, lat+0.01],\
              [lon-0.01, lat-0.01],\
              [lon+0.01, lat-0.01],\
              [lon+0.01, lat+0.01]])

    redband = satellite_database[satellite]['Red']
    greenband = satellite_database[satellite]['Green']
    blueband = satellite_database[satellite]['Blue']
    nirband = satellite_database[satellite]['NIR']
    swir1band = satellite_database[satellite]['SWIR1']
    swir2band = satellite_database[satellite]['SWIR2']
    cloudband = satellite_database[satellite]['Cloud']
    foldername = satellite_database[satellite]['Folder']

    # Pull the desired collection; filter date, region and bands
    # filterMetaData allows us to pick the desired Grid Reference System. Since the images appeared identicle, I picked 31SGR...
    # If the other MGRS is better, we can remove filterMetadata, reprint, and pick the other. 
    if satellite == 'Sentinel-2':
        _default_value = None
        scaleFac = 0.0001
        img_collection = get_s2_cld_col(region, startDate, endDate)
        if img_collection is not None:
            img_collection = img_collection.map(add_cloud_bands).select([redband,
                                            greenband,
                                            blueband,
                                            nirband,
                                            swir1band,
                                            swir2band,
                                            'cloud_prob'])
    else:
        _default_value = -999
        scaleFac = 1
        img_collection = ee.ImageCollection('LANDSAT/%s/C01/T1_RT_TOA'%foldername).filterDate(startDate, endDate).filterBounds(region).select([redband,
                                                                        greenband,
                                                                        blueband,
                                                                        nirband,
                                                                        swir1band,
                                                                        swir2band,
                                                                        cloudband])

    # initialize arrays
    chanlarr = None
    zarr = None
    lonarr = None
    latarr = None
    date_list2 = []
    u10m, v10m = [], []

    if img_collection is None:
        tkframe.post_print('>  ==>  !!!!! NO SATELLITE IMAGE FOUND !!!!!') 
        id_list, date_list = [], []
        pass

    tkframe.post_print('> Number of images found: '+ str(img_collection.size().getInfo()))
    tkframe.post_print('> ==> Zero img check: ' + str(img_collection.size().getInfo() == 0))

    # convert to list of images
    collectionList = img_collection.toList(img_collection.size())



    if img_collection.size().getInfo() == 0:
        tkframe.post_print('>  ==>  !!!!! NO SATELLITE IMAGE FOUND !!!!!') 
        id_list, date_list = [], []
        pass
    else:
        ### DATELIST for plumes ###
        methaneAlt = img_collection.getRegion(re,50).getInfo()
        id_list    = pd.DataFrame(methaneAlt)
        headers    = id_list.iloc[0]
        id_list    = pd.DataFrame(id_list.values[1:], columns=headers)                             
        id_list    = id_list[['id']].dropna().values.tolist()
        tkframe.post_print('>  ==> Image ids: ' + str(id_list))

        # Get the dates and format them
        if satellite == 'Sentinel-2':
            date_list = [x[0].split('_')[1].split('T')[0] for x in id_list] #FOR SENTINEL 2 AJT
            date_list = [datetime.strptime(x,'%Y%m%d').date().isoformat() for x in date_list]
        else:
            date_list = [x[0].split('_')[2] for x in id_list] #FOR LANDSAT 8
            date_list = [datetime.strptime(x,'%Y%m%d').date().isoformat() for x in date_list]
        tkframe.post_print('>  ==> Image dates: ' + str(date_list))


    for i in range(img_collection.size().getInfo()):
        try:
            tkframe.post_print('>  ==> Datetime now: ' + str(id_list[i]) + '  '+ str(date_list[i]))
        except:
            tkframe.post_print('>  ==> Datetime NA')
            id_list.append(None)
            date_list.append(None)
            pass

        try:
            currentimg = ee.Image(collectionList.get(i))
            imgdate = datetime(1970, 1, 1, 0, 0) + timedelta(seconds=currentimg.date().getInfo()['value']/1000)
            tkframe.post_print('>  ==> Img date: ' + imgdate.strftime('%Y-%m-%d %H:%M%S'))

            try:
                wind_collection = ee.ImageCollection("ECMWF/ERA5/DAILY").filterDate(imgdate.strftime('%Y-%m-%d')).filterBounds(era5_region).select(['u_component_of_wind_10m','v_component_of_wind_10m'])
                wind = wind_collection.first()
                u = geemap.ee_to_numpy(wind.select('u_component_of_wind_10m'), region = era5_region)/1.944 # convert m/s to knots
                v = geemap.ee_to_numpy(wind.select('v_component_of_wind_10m'), region = era5_region)/1.944 # convert m/s to knots
                u = np.nanmean(u)
                v = np.nanmean(v)
            except:
                tkframe.post_print('>  ==> ERA5 U/V winds NA')
                u10m.append(None)
                v10m.append(None)
                pass
            else:
                u10m.append(u)
                v10m.append(v)



            lons = currentimg.pixelLonLat().select('longitude').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            lats = currentimg.pixelLonLat().select('latitude').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            lons = np.squeeze(geemap.ee_to_numpy(lons, region=region))
            lats = np.squeeze(geemap.ee_to_numpy(lats, region=region))

            B6channel = currentimg.select(swir1band).multiply(scaleFac)
            B7channel = currentimg.select(swir2band).multiply(scaleFac)
            SWIR1img = B6channel.reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            SWIR2img = B7channel.reproject(crs=ee.Projection('EPSG:3395'), scale=30)

            # To numpy array
            SWIR1_geemap = geemap.ee_to_numpy(SWIR1img, region=region, default_value=_default_value)
            SWIR2_geemap = geemap.ee_to_numpy(SWIR2img, region=region, default_value=_default_value)

            if np.any(SWIR1_geemap == _default_value):
                SWIR1_geemap[np.where(SWIR1_geemap == _default_value)] = np.nan
            if np.any(SWIR2_geemap == _default_value):
                SWIR2_geemap[np.where(SWIR2_geemap == _default_value)] = np.nan

            SWIR1_flat = np.reshape(np.squeeze(SWIR1_geemap),-1)
            SWIR2_flat = np.reshape(np.squeeze(SWIR2_geemap),-1)

            mask = np.where(np.logical_and(~np.isnan(SWIR1_flat), ~np.isnan(SWIR2_flat)))
            SWIR1_flat = SWIR1_flat[mask]
            SWIR2_flat = SWIR2_flat[mask]

            SWIR1_flat = np.array(SWIR1_flat).reshape(-1,1)

            model = HuberRegressor().fit(SWIR1_flat, SWIR2_flat)
            b0 = 1/model.coef_[0] #This slope is SWIR2/SWIR1 

            dR = ee.Image(B6channel.multiply((b0)).subtract(B7channel).divide(B7channel)).rename('dR')
            dR = dR.reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            dR = np.squeeze(geemap.ee_to_numpy(dR, region=region, default_value=_default_value))
            dR[dR == _default_value] = np.nan

            if do_retrieval:
                test_retrieval = retrieve(dR, 'L8', method, targheight, obsheight, solarangle, obsangle, num_layers) ### retrieval
                z = test_retrieval*-1

            # get RGB, NIR, SWIRI, SWIRII channels from Landsat 8
            bchannel = currentimg.select(blueband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            bchannel = np.squeeze(geemap.ee_to_numpy(bchannel, region=region, default_value=_default_value))

            gchannel = currentimg.select(greenband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            gchannel = np.squeeze(geemap.ee_to_numpy(gchannel, region=region, default_value=_default_value))

            rchannel = currentimg.select(redband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            rchannel = np.squeeze(geemap.ee_to_numpy(rchannel, region=region, default_value=_default_value))

            nirchannel = currentimg.select(nirband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            nirchannel = np.squeeze(geemap.ee_to_numpy(nirchannel, region=region, default_value=_default_value))

            swir1channel = currentimg.select(swir1band).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            swir1channel = np.squeeze(geemap.ee_to_numpy(swir1channel, region=region, default_value=_default_value))

            swir2channel = currentimg.select(swir2band).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            swir2channel = np.squeeze(geemap.ee_to_numpy(swir2channel, region=region, default_value=_default_value))

            if satellite == 'Sentinel-2':
                cloudscore = currentimg.select('cloud_prob').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
                cloudscore = np.squeeze(geemap.ee_to_numpy(cloudscore, region=region, default_value=_default_value))
            else:
                cloudscore = ee.Algorithms.Landsat.simpleCloudScore(currentimg).select(['cloud'])
                cloudscore = cloudscore.reproject(crs=ee.Projection('EPSG:3395'), scale=30)
                cloudscore = np.squeeze(geemap.ee_to_numpy(cloudscore, region=region, default_value=100))

            if np.any(rchannel == _default_value):
                rchannel[rchannel == _default_value] = np.nan
            if np.any(gchannel == _default_value):
                gchannel[gchannel == _default_value] = np.nan
            if np.any(bchannel == _default_value):
                bchannel[bchannel == _default_value] = np.nan
            if np.any(nirchannel == _default_value):
                nirchannel[nirchannel == _default_value] = np.nan
            if np.any(swir1channel == _default_value):
                swir1channel[swir1channel == _default_value] = np.nan
            if np.any(swir2channel == _default_value):
                swir2channel[swir2channel == _default_value] = np.nan

            chanls = np.stack([dR, rchannel, gchannel, bchannel, nirchannel, swir1channel, swir2channel, cloudscore], axis=-1)
            date_list2.append(imgdate)

            if chanlarr is None:     # Initialize arrays
                # dRarr = dR[np.newaxis, :, :]
                chanlarr = chanls[np.newaxis, :, :, :]
                lonarr = lons[np.newaxis, :, :]
                latarr = lats[np.newaxis, :, :]
                if do_retrieval:
                    zarr = z[np.newaxis, :, :]
                else:
                    zarr = np.array([None])
            else:
                # dRarr = np.concatenate((dRarr, dR[np.newaxis, :, :]), axis=0)
                chanlarr = np.concatenate((chanlarr, chanls[np.newaxis, :, :, :]), axis=0)
                lonarr = np.concatenate((lonarr, lons[np.newaxis, :, :]), axis=0)
                latarr = np.concatenate((latarr, lats[np.newaxis, :, :]), axis=0)
                if do_retrieval:
                    zarr = np.concatenate((zarr, z[np.newaxis, :, :]), axis=0)
                else:
                    zarr = np.append(zarr, None)
        except Exception as e:
            tkframe.post_print(">  ==> !!!Something went wrong!!!: " + str(e))
            pass
    return id_list, date_list, date_list2, chanlarr, zarr, lonarr, latarr, u10m, v10m


