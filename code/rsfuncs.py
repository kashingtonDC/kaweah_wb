'''
Aakash Ahamed
Stanford University dept of Geophysics 
aahamed@stanford.edu 

Codes to process geospatial data in earth engine and python 
'''

import os
import ee
import time
import tqdm
import fiona
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import geopandas as gp

from osgeo import gdal
from osgeo import osr
from datetime import timedelta
from rasterio import features, mask
from shapely.ops import unary_union
from climata.usgs import DailyValueIO
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

from tqdm import tqdm


''' 
#############################################################################################################

Helpers for working with dataframes, times, dicts etc

#############################################################################################################
'''


def col_to_dt(df):
	'''
	converts the first col of a dataframe read from CSV to datetime
	'''
	t = df.copy()
	t['dt'] = pd.to_datetime(df[df.columns[0]])
	t = t.set_index(pd.to_datetime(t[t.columns[0]]))
	t.drop([t.columns[0], "dt"],axis = 1, inplace = True)
	
	return t

def dl_2_df(dict_list, dt_idx):
	'''
	converts a list of dictionaries to a single dataframe
	'''
	alldat = [item for sublist in [x.values() for x in dict_list] for item in sublist]
	# Make the df
	alldata = pd.DataFrame(alldat).T
	alldata.index = dt_idx
	col_headers = [item for sublist in [x.keys() for x in dict_list] for item in sublist]
	alldata.columns = col_headers

	return alldata


''' 
#############################################################################################################

Vector / Shapefile / EE Geometry Functions

#############################################################################################################
'''

# TODO: Make a single function that converts shp (multipoly / single poly / points / ) --> EE geom

def gdf_to_ee_poly(gdf, simplify = True):

	if simplify:
		gdf = gdf.geometry.simplify(0.01)
	
	lls = gdf.geometry.iloc[0]
	x,y = lls.exterior.coords.xy
	coords = [list(zip(x,y))]
	area = ee.Geometry.Polygon(coords)

	return area

def gdf_to_ee_multipoly(gdf):

	lls = gdf.geometry.iloc[0]
	mps = [x for x in lls]
	multipoly = []

	for i in mps: 
		x,y = i.exterior.coords.xy
		coords = [list(zip(x,y))]
		multipoly.append(coords)

	return ee.Geometry.MultiPolygon(multipoly)

def get_area(gdf, fast = True):

	t = gdf.buffer(0.001).unary_union
	d  = gp.GeoDataFrame(geometry=gp.GeoSeries(t))
	if fast:
		d2  = gp.GeoDataFrame(geometry=gp.GeoSeries(d.simplify(0.001))) 
		area = gdf_to_ee_multipoly(d2)
	else:
		area = gdf_to_ee_multipoly(d)
		
	return area

def gen_polys(geometry, dx=0.5, dy=0.5):
	
	'''
	Input: ee.Geometry
	Return: ee.ImaceCollection of polygons 
	Use: Subpolys used to submit full res (30m landsat; 10m sentinel) resolution for large areas 
	'''
	
	bounds = ee.Geometry(geometry).bounds()
	coords = ee.List(bounds.coordinates().get(0))
	ll = ee.List(coords.get(0))
	ur = ee.List(coords.get(2))
	xmin = ll.get(0)
	xmax = ur.get(0)
	ymin = ll.get(1)
	ymax = ur.get(1)

	xx = ee.List.sequence(xmin, xmax, dx)
	yy = ee.List.sequence(ymin, ymax, dy)
	
	polys = []

	for x in tqdm(xx.getInfo()):
		for y in yy.getInfo():
			x1 = ee.Number(x).subtract(ee.Number(dx).multiply(0.5))
			x2 = ee.Number(x).add(ee.Number(dx).multiply(0.5))
			y1 = ee.Number(y).subtract(ee.Number(dy).multiply(0.5))
			y2 = ee.Number(y).add(ee.Number(dy).multiply(0.5))
			geomcoords = ee.List([x1, y1, x2, y2]);
			rect = ee.Algorithms.GeometryConstructors.Rectangle(geomcoords);
			polys.append(ee.Feature(rect))

	return ee.FeatureCollection(ee.List(polys)).filterBounds(geometry)
  


''' 
#############################################################################################################

Matplotlib Plotting for Vectors / Shapefiles

#############################################################################################################
'''


def draw_poly(gdf, mpl_map, facecolor = "red",  alpha = 0.3, edgecolor = 'black', lw = 1, fill = True):
	'''
	Turns a geopandas gdf into matplotlib polygon patches for friendly plotting with basemap. 
	
	'''    

	for index, row in gdf.iterrows():
		lats = []
		lons = []
		for pt in list(row['geometry'].exterior.coords): 
			lats.append(pt[1])
			lons.append(pt[0])

		x, y = m( lons, lats )
		xy = zip(x,y)
		poly = Polygon(list(xy), fc=facecolor, alpha=alpha, ec = edgecolor ,lw = lw, fill = fill)
		plt.gca().add_patch(poly)

	return

def draw_polys(gdf, mpl_map, facecolor = "red",  alpha = 0.3, edgecolor = 'black', lw = 1, fill = True, zorder = 3):
	'''
	Turns a geopandas gdf of multipolygons into matplotlib polygon patches for friendly plotting with basemap. 
	'''
	
	for index, row in gdf.iterrows():
		lats = []
		lons = []
		for pt in list(row['geometry'].exterior.coords): 
			lats.append(pt[1])
			lons.append(pt[0])

		x, y = m( lons, lats )
		xy = zip(x,y)
		poly = Polygon(list(xy), fc=facecolor, alpha=alpha, ec = edgecolor ,lw = lw, fill = fill, zorder = zorder)
		plt.gca().add_patch(poly)

	return


def draw_points(gdf, mpl_map, sizecol = None, color = 'red', alpha = 0.7, edgecolor = None, fill = True, zorder = 4):
	'''
	Turns a geopandas gdf of points into matplotlib lat/lon objects for friendly plotting with basemap. 
	'''
	lats = []
	lons = []
	for index, row in gdf.iterrows():
		for pt in list(row['geometry'].coords): 
			lats.append(pt[1])
			lons.append(pt[0])
	
	if sizecol is None:
		sizecol = 50
	else:
		sizecol = sizecol.values
	
	mpl_map.scatter(lons, lats, latlon=True, s = sizecol, alpha=alpha, c = color, edgecolor = edgecolor, zorder = zorder)

	return

''' 
#############################################################################################################

EE Wrappers

#############################################################################################################
'''

def calc_monthly_sum(dataset, startdate, enddate, area):
	'''
	Calculates monthly sums (pd.Dataframe) for EE data given startdate, enddate, and area
	Datasets are stored in `data` dict below.
	Note the "scaling_factor" parameter, 
	which is provided by EE for each dataset, and further scaled by temporal resolution to achieve monthly resolution
	This is explicitly written in the `data` dict 
	
	EE will throw a cryptic error if the daterange you input is not valid for the product of interest. 
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	resolution = dataset[3]
	
	dt_idx = pd.date_range(startdate,enddate, freq='MS')
	sums = []
	seq = ee.List.sequence(0, len(dt_idx)-1)
	num_steps = seq.getInfo()

	print("processing:")
	print("{}".format(ImageCollection.first().getInfo()['id']))

	for i in tqdm(num_steps):

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		im = ee.Image(ImageCollection.select(var).filterDate(start, end).sum().set('system:time_start', start.millis()))
		scale = im.projection().nominalScale()
		scaled_im = im.multiply(scaling_factor).multiply(ee.Image.pixelArea()).multiply(1e-12) # mm --> km^3
		
		sumdict  = scaled_im.reduceRegion(
			reducer = ee.Reducer.sum(),
			geometry = area,
			scale = resolution,
			bestEffort= True)

		total = sumdict.getInfo()[var]
		sums.append(total)
		
	sumdf = pd.DataFrame(np.array(sums), dt_idx + MonthEnd(0))
	sumdf.columns = [var]
	df = sumdf.astype(float)
		
	return df

def calc_monthly_mean(dataset, startdate, enddate, area):
	'''
	Same as above, but calculates mean (useful for anoamly detection,  state variables like SM and SWE)
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	
	dt_idx = pd.date_range(startdate,enddate, freq='MS')
	sums = []
	seq = ee.List.sequence(0, len(dt_idx)-1)
	num_steps = seq.getInfo()

	print("processing:")
	print("{}".format(ImageCollection.first().getInfo()['id']))

	for i in tqdm(num_steps):

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).mean().set('system:time_start', start.millis())
		scale = im.projection().nominalScale()
		scaled_im = im.multiply(scaling_factor).multiply(ee.Image.pixelArea()).multiply(1e-12) # mm --> km^3
		
		sumdict  = scaled_im.reduceRegion(
			reducer = ee.Reducer.sum(),
			geometry = area,
			scale = scale,
			bestEffort = True)

		total = sumdict.getInfo()[var]
		sums.append(total)
		
	sumdf = pd.DataFrame(np.array(sums), dt_idx + MonthEnd(0))
	sumdf.columns = [var]
	df = sumdf.astype(float)
		
	return df

def get_grace(dataset, startdate, enddate, area):
	'''
	Get Grace data from EE. Similar to above 
	'''

	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	
	dt_idx = pd.date_range(startdate,enddate, freq='M')
	
	sums = []
	seq = ee.List.sequence(0, len(dt_idx)-1)
	
	print("processing:")
	print("{}".format(ImageCollection.first().getInfo()['id']))
	
	num_steps = seq.getInfo()

	for i in tqdm(num_steps):

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		try:
			im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).sum().set('system:time_start', end.millis())
			t2 = im.multiply(ee.Image.pixelArea()).multiply(scaling_factor).multiply(1e-6) # Multiply by pixel area in km^2

			scale = t2.projection().nominalScale()
			sumdict  = t2.reduceRegion(
					reducer = ee.Reducer.sum(),
					geometry = area,
					scale = scale)

			result = sumdict.getInfo()[var] * 1e-5 # cm to km
			sums.append(result)
		except:
			sums.append(np.nan) # If there is no grace data that month, append a np.nan 

	sumdf = pd.DataFrame(np.array(sums), dt_idx)
	sumdf.columns = [var]
	df = sumdf.astype(float)

	return df

def get_ims(dataset, startdate,enddate, area, return_dates = False, table = False, monthly_mean = False,  monthly_sum = False):
	
	'''
	Returns gridded images for EE datasets 
	'''

	if monthly_mean:
		if monthly_sum:
			raise ValueError("cannot perform mean and sum reduction at the same time")              

	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	native_res = dataset[3]

	dt_idx = pd.date_range(startdate,enddate, freq='MS')
	ims = []
	seq = ee.List.sequence(0, len(dt_idx)-1)
	num_steps = seq.getInfo()

	# TODO: Make this one loop ?

	print("processing:")
	print("{}".format(ImageCollection.first().getInfo()['id']))

	for i in tqdm(num_steps):

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		if monthly_mean:
			im1 = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).mean().set('system:time_start', end.millis())
			im = ee.ImageCollection(im1)
		elif monthly_sum:
			im1 = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).sum().set('system:time_start', end.millis())
			im = ee.ImageCollection(im1)
		else:
			im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).set('system:time_start', end.millis())
		
		# This try / catch is probably not great, but needs to be done for e.g. grace which is missing random months 
		try:
			result = im.getRegion(area,native_res,"epsg:4326").getInfo()
			ims.append(result)
		except:
			continue
		

	results = []
	dates = []

	print("postprocesing")

	for im in tqdm(ims):
		header, data = im[0], im[1:]

		df = pd.DataFrame(np.column_stack(data).T, columns = header)
		df.latitude = pd.to_numeric(df.latitude)
		df.longitude = pd.to_numeric(df.longitude)
		df[var] = pd.to_numeric(df[var])

		if table:
			results.append(df)
			continue

		images = []

		for idx,i in enumerate(df.id.unique()):

			t1 = df[df.id==i]
			arr = array_from_df(t1,var)
			arr[arr == 0] = np.nan
			images.append(arr*scaling_factor)# This is the only good place to apply the scaling factor. 

			if return_dates:
				date = df.time.iloc[idx]
				dates.append(datetime.datetime.fromtimestamp(date/1000.0))

		results.append(images) 

	print("====COMPLETE=====")

	# Unpack the list of results 
	if return_dates:
		return [ [item for sublist in results for item in sublist], dates]
	else:   
		return [item for sublist in results for item in sublist] 



def array_from_df(df, variable):    

	'''
	Convets a pandas df with lat, lon, variable to a numpy array 
	'''

	# get data from df as arrays
	lons = np.array(df.longitude)
	lats = np.array(df.latitude)
	data = np.array(df[variable]) # Set var here 
											  
	# get the unique coordinates
	uniqueLats = np.unique(lats)
	uniqueLons = np.unique(lons)

	# get number of columns and rows from coordinates
	ncols = len(uniqueLons)    
	nrows = len(uniqueLats)

	# determine pixelsizes
	ys = uniqueLats[1] - uniqueLats[0] 
	xs = uniqueLons[1] - uniqueLons[0]

	# create an array with dimensions of image
	arr = np.zeros([nrows, ncols], np.float32)

	# fill the array with values
	counter =0
	for y in range(0,len(arr),1):
		for x in range(0,len(arr[0]),1):
			if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
				counter+=1
				arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
	
	return arr




# This is the staging area. Haven's used these in a while, or not tested altogether. 

def img_to_arr(eeImage, var_name, area, scale = 30):
	temp = eeImage.select(var_name).clip(area)
	latlon = eeImage.pixelLonLat().addBands(temp)
	
	latlon = latlon.reduceRegion(
		reducer = ee.Reducer.toList(),
		geometry = area, 
		scale = scale
		)
	
	data = np.array((ee.Array(latlon.get(var_name)).getInfo()))
	lats = np.array((ee.Array(latlon.get('latitude')).getInfo()))
	lons = np.array((ee.Array(latlon.get('longitude')).getInfo()))
	
	lc,freq = np.unique(data,return_counts = True)
	
	return data, lats,lons 

def imc_to_arr(eeImage):
	temp = eeImage.filterBounds(area).first().pixelLonLat()
	
	latlon = temp.reduceRegion(
		reducer = ee.Reducer.toList(),
		geometry = area, 
		scale = 1000
		)
	
	data = np.array((ee.Array(latlon.get('cropland')).getInfo()))
	lats = np.array((ee.Array(latlon.get('latitude')).getInfo()))
	lons = np.array((ee.Array(latlon.get('longitude')).getInfo()))
	
	lc,freq = np.unique(data,return_counts = True)
	
	return data, lats,lons 

def arr_to_img(data,lats,lons):
	uniquelats = np.unique(lats)
	uniquelons = np.unique(lons)
	
	ncols = len(uniquelons)
	nrows = len(uniquelats)
	
	ys = uniquelats[1] - uniquelats[0]
	xs = uniquelons[1] - uniquelons[0]
	
	arr = np.zeros([nrows, ncols], np.float32)
	
	counter = 0
	for y in range(0, len(arr),1):
		for x in range(0, len(arr[0]),1):
			if lats[counter] == uniquelats[y] and lons[counter] == uniquelons[x] and counter < len(lats)-1:
				counter+=1
				arr[len(uniquelats)-1-y,x] = data[counter]
				
	return arr

def freq_hist(eeImage, area, scale, var_name):    
	freq_dict = ee.Dictionary(
	  eeImage.reduceRegion(ee.Reducer.frequencyHistogram(), area, scale).get(var_name)
	);
	
	return freq_dict

''' 
#############################################################################################################

NetCDF / Gtiff Functions

#############################################################################################################
'''

def get_lrm_swe(shppath, data_dir ="../data/LRM_SWE_monthly" ):
    '''
    Given a path to a shapefile, compute the monthly SWE
    Input: (str) - path to shapefile
    Output: (pd.DataFrame) - monthly SWE 
    '''
    
    # Find SWE files
    files = [os.path.join(data_dir,x) for x in os.listdir(data_dir) if x.endswith(".tif")]
    files.sort()

    # Read CVWS shapefile
    with fiona.open(shppath, "r") as shapefile:
        cvws_geom = [feature["geometry"] for feature in shapefile]

    # Read the files, mask nans, clip to CVWS, extract dates
    imdict = {}

    for i in tqdm(files[:]):
        date = datetime.datetime.strptime(i[-12:-4],'%Y%m%d')+ timedelta(days=-1) # Get the date 
        datestr = date.strftime('%Y%m%d') # Format date
        src = rio.open(i) # Read file
        src2 = rio.mask.mask(src, cvws_geom, crop=True) # Clip to shp 
        arr = src2[0] # read as array
        arr = arr.reshape(arr.shape[1], arr.shape[2]) # Reshape bc rasterio has a different dim ordering 
        arr[arr < 0 ] = np.nan # Mask nodata vals 
        imdict[datestr] = arr
        
    # Fill in the dates with no SWE with nans 
    dt_idx = pd.date_range(list(imdict.keys())[0], list(imdict.keys())[-1], freq = "M")

    all_dates = {}

    for i in dt_idx:
        date = i.strftime("%Y%m%d") 

        if date in imdict.keys():
            im = imdict[date]
        else:
            im = np.zeros_like(list(imdict.values())[0])
            im[im==0] = np.nan
        all_dates[date] = im

    # Stack all dates to 3D array
    cvws_swe = np.dstack(list(all_dates.values()))

    # Compute monthly sums
    swesums = []
    for i in range(cvws_swe.shape[2]):
        swesums.append(np.nansum(cvws_swe[:,:,i] *500**2 * 1e-9)) # mult by 2500m pixel area, convert m^3 to km^3

    swedf = pd.DataFrame(swesums,dt_idx)
    swedf.columns = ['swe_lrm']
    return swedf

def get_snodas_swe(shppath, data_dir ="/Users/aakash/Desktop/SatDat/SNODAS/SNODAS_CA_processed/" ):
    '''
    Given a path to a shapefile, compute the monthly SWE
    Input: (str) - path to shapefile
    Output: (pd.DataFrame) - monthly SWE 
    '''
    
    # Find SWE files
    files = [os.path.join(data_dir,x) for x in os.listdir(data_dir) if x.endswith(".tif")]
    files.sort()

    # Read CVWS shapefile
    with fiona.open(shppath, "r") as shapefile:
        cvws_geom = [feature["geometry"] for feature in shapefile]

    # Read the files, mask nans, clip to CVWS, extract dates
    imdict = {}

    for i in tqdm(files[:]):
        date = datetime.datetime.strptime(i[-16:-8],'%Y%m%d')# Get the date 
        if date.day == 1:
            datestr = date.strftime('%Y%m%d') # Format date
            src = rio.open(i) # Read file
            src2 = rio.mask.mask(src, cvws_geom, crop=True) # Clip to shp 
            arr = src2[0].astype(float) # read as array
            arr = arr.reshape(arr.shape[1], arr.shape[2]) # Reshape bc rasterio has a different dim ordering 
            arr[arr < 0 ] = np.nan # Mask nodata vals 
            imdict[datestr] = arr/1000 # divide by scale factor to get SWE in m 
        
    # Stack all dates to 3D array
    cvws_swe = np.dstack(list(imdict.values()))

    # Compute monthly sums
    swesums = []
    for i in range(cvws_swe.shape[2]):
        swesums.append(np.nansum(cvws_swe[:,:,i] * 1000**2 * 1e-9)) # multiply by 1000m pixel size, convert m^3 to km^3

    dt_idx = [datetime.datetime.strptime(x, '%Y%m%d')+ timedelta(days=-1)  for x in imdict.keys()]
    swedf = pd.DataFrame(swesums,dt_idx)
    swedf.columns = ['swe_snodas']
    return swedf

def get_ssebop(shppath):
    '''
    Given a path to a shapefile, compute the monthly SSEBop ET
    Input: (str) - path to shapefile
    Output: (pd.DataFrame) - monthly SSEBop ET  
    '''
    
    files = [os.path.join("../data",x) for x in os.listdir("../data") if x.endswith("nc") if "SSEBOP" in x]
    ds = xr.open_dataset(files[0])
    
    gdf = gp.read_file(shppath)
    
    ds['catch'] = rasterize(gdf.geometry, ds['et'][0].coords)
    ssebop_masked = ds['et']*ds['catch']
    
    dt = pd.date_range(ds.time[0].values, ds.time[-1].values, freq = "MS")

    et= []
    for i in ssebop_masked:
        et.append(i.sum()* 1e-6) # m^2 to km^2 
        
    etdf = pd.DataFrame({'et': np.array(et)}, index = dt)
    etdf.columns = ["aet_ssebop"]
    etdf.set_index(etdf.index + MonthEnd(0), inplace = True)
    
    return etdf
    

def transform_from_latlon(lat, lon):
	lat = np.asarray(lat)
	lon = np.asarray(lon)
	trans = Affine.translation(lon[0], lat[0])
	scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
	return trans * scale

def rasterize(shapes, coords, fill=np.nan, **kwargs):
	"""
	Rasterize a list of (geometry, fill_value) tuples onto the given
	xray coordinates. This only works for 1d latitude and longitude
	arrays.
	"""
	transform = transform_from_latlon(coords['lat'], coords['lon'])
	out_shape = (len(coords['lat']), len(coords['lon']))
	raster = features.rasterize(shapes, out_shape=out_shape,
								fill=fill, transform=transform,
								dtype=float, **kwargs)
	return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))


def write_raster(array,gdf,outfn):
	'''
	converts a numpy array and a geopandas gdf to a geotiff
	Data values are stored in np.array
	spatial coordinates stored in gdf
	outfn - outpath
	'''
	
	xmin, ymin = gdf.bounds.minx.values[0], gdf.bounds.miny.values[0]
	xmax, ymax = gdf.bounds.maxx.values[0], gdf.bounds.maxy.values[0]
	nrows, ncols = array.shape
	xres = (xmax-xmin)/float(ncols)
	yres = (ymax-ymin)/float(nrows)
	geotransform =(xmin,xres,0,ymax,0, -yres)   

	output_raster = gdal.GetDriverByName('GTiff').Create(outfn,ncols, nrows, 1 , gdal.GDT_Float32)  # Open the file
	output_raster.SetGeoTransform(geotransform)  # Specify coords
	srs = osr.SpatialReference()                 # Establish encoding
	srs.ImportFromEPSG(4326)                     # WGS84 lat long
	output_raster.SetProjection(srs.ExportToWkt() )   # Export coordinate system 
	output_raster.GetRasterBand(1).WriteArray(array)   # Write array to raster
	
	print("wrote {}".format(outfn))
	return outfn

''' 
#############################################################################################################

EE Datasets

#############################################################################################################
'''

def load_data():

	'''
	This data structure has the following schema:

	data (dict)
	keys: {product}_{variable}
	values: 
	(1) ImageColection
	(2) variable name
	(3) scale factor - needed to calculate volumes when computing sums. Depends on units and sampling frequency 
	(4) native resolution - needed to return gridded images 


	'''
	data = {}

	###################
	##### ET data #####
	###################

	# https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2
	data['modis_aet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "ET", 0.1, 1000]
	data['modis_pet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "PET", 0.1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H
	data['gldas_aet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'Evap_tavg', 86400*30 / 240 , 25000]   # kg/m2/s --> km3 / mon , noting 3 hrly images
	data['gldas_pet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'PotEvap_tavg', 1 / 240, 25000] 

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_NLDAS_FORA0125_H002
	data['nldas_pet'] = [ee.ImageCollection('NASA/NLDAS/FORA0125_H002'), 'potential_evaporation', 1, 12500]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
	data['tc_aet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "aet", 0.1 , 1000]
	data['tc_pet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "pet", 0.1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
	data['gmet_etr'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "etr", 1 , 1000]
	data['gmet_eto'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "eto", 1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001
	data['fldas_aet'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "Evap_tavg", 86400*30, 12500]

	###################
	##### P data ######
	###################

	data['trmm']  =  [ee.ImageCollection('TRMM/3B43V7'), "precipitation", 720, 25000] # scale hours per month
	data['prism'] = [ee.ImageCollection("OREGONSTATE/PRISM/AN81m"), "ppt", 1, 4000]
	data['chirps'] = [ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD'), "precipitation", 1, 5500]
	data['persiann'] = [ee.ImageCollection("NOAA/PERSIANN-CDR"), "precipitation", 1, 25000]
	data['dmet'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V4'), "prcp", 1, 4000]
	data['gpm'] = [ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06"), "precipitation", 720, 12500] # scale hours per month

	#################### 
	##### SWE data #####
	####################
	data['fldas_swe'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "SWE_inst", 1 , 12500]
	data['gldas_swe'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "SWE_inst", 1 , 25000]
	data['dmet_swe'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V4'), "swe", 1, 4000] # Reduced from 1000 because the query times out over the whole CVW 
	data['tc_swe'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "swe", 1, 4000]

	####################
	##### R data #######
	####################
	data['tc_r'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "ro", 1, 4000]
	
	# FlDAS
	data['fldas_ssr'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "Qs_tavg", 86400*30, 12500] # kg/m2/s --> km3 / mon
	data['fldas_bfr'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "Qsb_tavg", 86400*30, 12500]

	# GLDAS
	data['gldas_ssr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qs_acc", 1, 25000]
	data['gldas_bfr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsb_acc", 1, 25000 ]
	data['gldas_qsm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsm_acc", 1, 25000]

	# ECMWF
	data['ecmwf_r'] = [ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") , 'runoff', 3e4, 11132] # m/d --> mm/mon
	
	#####################
	##### SM data #######
	#####################
	data['tc_sm'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "soil", 0.1, 4000]

	data['fsm1'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi00_10cm_tavg", 86400*24 , 12500]
	data['fsm2'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi10_40cm_tavg", 86400*24 , 12500]
	data['fsm3'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi40_100cm_tavg", 86400*24 , 12500]
	data['fsm4'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi100_200cm_tavg", 86400*24 , 12500]

	data['gldas_rzsm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "RootMoist_inst", 1, 25000]

	data['gsm1'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi0_10cm_inst", 1 ,25000]
	data['gsm2'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi10_40cm_inst", 1 ,25000]
	data['gsm3'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi40_100cm_inst", 1 ,25000]
	data['gsm4'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi100_200cm_inst", 1 ,25000]

	data['smap_ssm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture") , "ssm", 1 ,10000]
	data['smap_susm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture"), "susm", 1 ,10000]
	data['smap_smp'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture"), "smp", 1 ,10000]

	data['smos_ssm'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "ssm", 1 ,25000]
	data['smos_susm'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "susm", 1 ,25000]
	data['smos_smp'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "smp", 1 ,25000]
	############################
	##### Elevation data #######
	############################

	data['srtm'] = [ee.Image("CGIAR/SRTM90_V4"), "elevation", 1 ,1000]

	#########################
	##### Gravity data ######
	#########################
	data['jpl'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_jpl",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]
	data['csr'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_csr",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]
	data['gfz'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_gfz",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]

	data['mas'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "lwe_thickness", 1] 
	data['mas_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "uncertainty", 1] 

	data['cri'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "lwe_thickness", 1] 
	data['cri_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "uncerrtainty", 1] 


	#########################
	##### Optical data ######
	#########################

	data['modis_snow'] = [ee.ImageCollection('MODIS/006/MOD10A1'), "NDSI_Snow_Cover",  1, 2500] # reduced resolution  
	data['modis_ndvi'] = [ee.ImageCollection('MODIS/MCD43A4_NDVI'), "NDVI",  1, 500] 

	data['landsat_8_b1'] = [ee.ImageCollection('LANDSAT/LC08/C01/T1_SR'), "B1" ,  0.001, 30] 

	data['l8_ndwi_32d'] = [ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDWI'), "NDWI", 1, 30]
	data['l8_ndwi_annual'] = [ee.ImageCollection("LANDSAT/LC08/C01/T1_ANNUAL_NDWI"), "NDWI", 1, 30]


	###########################
	##### Landcover data ######
	###########################
	data['cdl'] = [ee.ImageCollection('USDA/NASS/CDL'), "cropland",  1, 30]
	data['nlcd'] = [ee.ImageCollection('USGS/NLCD'), 'landcover', 1, 30]

	return data



''' 
#############################################################################################################

Lookup tables

#############################################################################################################
'''


def cdl_2_faunt():
	
	'''
	Classify crop types from CDL to the faunt (2009), schmid (2004) scheme 

	CDL classes: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
	Faunt kc and classes: https://water.usgs.gov/GIS/metadata/usgswrd/XML/pp1766_fmp_parameters.xml 

	Dict Key is the Faunt class (int)     
	Dict Value is the CDL category (string)

	The faunt class = CDL category is shown at the top of each k:v pair. 
	'''
	
	data = {
		# Water = water(83), wetlands(87), Aquaculture(92), Open Water(111), Perreniel Ice / Snow (112)
		1 : ["83", "87", "92", "111", "112"], 
		# Urban = developed high intensity(124), developed medium intensity(123)
		2 : ["124", "123"], 
		# Native = grassland/pasture(176), Forest(63), Shrubs(64), barren(65, 131), Clover/Wildflowers(58)
		# Forests (141 - 143), Shrubland (152), Woody Wetlands (190), Herbaceous wetlands (195)
		3 : ["176","63","64", "65", "131","58", "141", "142", "143", "152", "190", "195"], 
		# Orchards, groves, vineyards = 
		4 : [""],
		# Pasture / hay = other hay / non alfalfa (37)
		5 : ["37"],
		# Row Crops = corn (1), soybeans (5),Sunflower(6) sweet corn (12), pop corn (13), double winter/corn (225), 
		# double oats/corn(226), double barley/corn(237), double corn / soybeans
		6 : ["1", "5", "6", "12", "13", "225", "226", "237", "239"] ,
		# Small Grains = Spring wheat (23), winter wheat (24), other small grains (25), winter wheat / soybeans (26), 
		# rye (27), oats (28), Millet(29), dbl soybeans/oats(240)
		7 : ["23", "24", "25", "26", "27", "28", "29", "240"] ,
		# Idle/fallow = Sod/Grass Seed (59), Fallow/Idle Cropland(61), 
		8 : ["59","61"],
		# Truck, nursery, and berry crops = 
		# Blueberries (242), Cabbage(243), Cauliflower(244), celery (245), radishes (246), Turnips(247)
		# Eggplants (249), Cranberries (250), Caneberries (55), Brocolli (214), Peppers(216), 
		# Greens(219), Strawberries (221), Lettuce (227), Double Lettuce/Grain (230 - 233)
		9 : ["242", "243", "244", "245", "246", "247", "248", "249", "250", "55", "214", "216","219","221", "227", "230", "231", "232", "233"], 

		# Citrus and subtropical = Citrus(72), Oranges (212), Pommegranates(217)
		10 : ["72", "212", "217"] ,

		# Field Crops = 
		# Peanuts(10),Mint (14),Canola (31),  Vetch(224),  Safflower(33) , RapeSeed(34), 
		# Mustard(35) Alfalfa (36),Camelina (38), Buckwheat (39), Sugarbeet (41), Dry beans (42), Potaoes (43)
		# Sweet potatoes(46), Misc Vegs & Fruits (47), Cucumbers(50)
		# Chick Peas(51),Lentils(52),Peas(53),Tomatoes(54)Hops(56),Herbs(57),Carrots(206),
		# Asparagus(207),Garlic(208), Cantaloupes(209), Honeydew Melons (213), Squash(222), Pumpkins(229), 

		11 : ["10",  "14", "224", "31","33", "34", "35", "36", "38", "39", "41", "42", "43", "46", "47", "48" ,
			  "49", "50", "51", "52", "53", "54",  "56", "57","206","207", "208", "209","213","222", "229"] ,

		# Vineyards = Grapes(69)
		12 : ["69"],
		# Pasture = Switchgrass(60)
		13 : ["60"],
		# Grain and hay = Sorghum(4), barley (21), Durham wheat (22), Triticale (205), 
		# Dbl grain / sorghum (234 - 236), Dbl 
		14 : ["4", "21", "22", "205", "234", "235", "236"],
		# livestock feedlots, diaries, poultry farms = 
		15 : [""],

		# Deciduous fruits and nuts = Pecans(74), Almonds(75), 
		# Walnuts(76), Cherries (66), Pears(77), Apricots (223), Apples (68), Christmas Trees(70)
		# Prunes (210), Plums (220), Peaches(67), Other Tree Crops (71), Pistachios(204), 
		# Olives(211), Nectarines(218), Avocado (215)
		16 : ["74", "75", "76","66","77", "223", "68", "210", "220", "67", "70", "71", "204", "211","215","218"],

		# Rice = Rice(3)
		17 : ["3"],
		# Cotton = Cotton (2) , Dbl grain / cotton (238-239)
		18 : ["2", "238", "239"], 
		# Developed = Developed low intensity (122) developed open space(121)
		19 : ["122", "121"],
		# Cropland and Pasture
		20 : [""],
		# Cropland = Other crops (44)
		21 : ["44"], 
		# Irrigated row and field crops = Woody Wetlands (190), Herbaceous wetlands(195)
		22 : [""] # ["190", "195"] 
		}
		
	return data



def nlcd_nums2classes(): 
	'''
	lookup table mapping numeric classes to nlcd labels 
	'''
	data = {11: 'Open Water',
			12: 'Perennial Ice/Snow',
			21: 'Developed, Open Space',
			22: 'Developed, Low Intensity',
			23: 'Developed, Medium Intensity',
			24: 'Developed, High Intensity',
			31: 'Barren Land (Rock/Sand/Clay)',
			41: 'Deciduous Forest ',
			42: 'Evergreen Forest',
			43: 'Mixed Forest',
			51: 'Dwarf Scrub',
			52: 'Shrub/Scrub',
			71: 'Grassland/Herbaceous',
			72: 'Sedge/Herbaceous',
			73: 'Lichens',
			74: 'Moss',
			81: 'Pasture/Hay',
			82: 'Cultivated Crops',
			90: 'Woody Wetlands',
			95: 'Emergent Herbaceous Wetlands'}
	return data 