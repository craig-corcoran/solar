import plac
import glob
import copy
import numpy
import pandas
import itertools as it
import netCDF4 as ncdf
import matplotlib.pyplot as pyplot
from scipy import interpolate

# XXX whats the difference between missing and mask?
# XXX separate into samples 
# XXX serialize samples or frames
# XXX store non-spatial parameters separately?
# XXX weight outliers lower during interpolation

class GoesData(object):
    
    def __init__(self, path, inputs, lat_range = None, lon_range = None, 
                interp_buffer = None, use_masked = True):
        
        self.inputs = inputs
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.interp_buffer = interp_buffer
        
        # read netcdf file into pandas data frame
        self.frame, self.meta = self.nc_to_frame(path, use_masked = use_masked) # removes all but lat, lon, inputs, and datetime

        # add datetime column
        self.frame['datetime'] = self.convert_to_datetime(
                                            self.frame['img_date'].values,
                                            self.frame['img_time'].values)
        assert (self.frame['datetime'].values == self.frame['datetime'].values[0]).all() # XXX
        self.timestamp = self.frame['datetime'].iloc[0]#[:13]
        #self.frame = self.frame.sort('datetime')
        
        print 'rescaling data'
        self.rescale()
        

    def rescale(self):
        '''rescale by units in metadata, center and set variance to unity, 
        stores mean and std for applying the same scaling or un-scaling later) 
        '''

        # convert lat to 0 to pi and lon to 0 to 2pi
        self.frame['lat_rad'], self.frame['lon_rad'] = self.latlon_to_thetaphi(
                                                        self.frame['lat_cell'], 
                                                        self.frame['lon_cell'])

        # scale inputs to native units
        for inp in self.inputs:
            mvar = self.meta.loc[inp]
            if not numpy.isnan(mvar['SCALED_MIN']): # if var was scaled
                var = self.frame[inp].astype(float) - mvar['SCALED_MIN']
                scal = float(mvar['RANGE_MAX'] - mvar['RANGE_MIN']) / float(mvar['SCALED_MAX'] - mvar['SCALED_MIN'])
                self.frame[inp] = (var * scal) + mvar['RANGE_MIN']
        
        # center and set var/std to 1
        #self.mean_vec = self.frame[self.inputs].mean()
        #self.std_vec = self.frame[self.inputs].std()

        #print 'mean: ', numpy.array(self.mean_vec)
        #print 'std: ', numpy.array(self.std_vec)

        #self.frame[self.inputs] = self.frame[self.inputs].astype(float) - self.mean_vec # XXX where to normalize data?
        #self.frame[self.inputs] = self.frame[self.inputs].astype(float) / self.std_vec

    # XXX unscale
    # convert theta/phi to lat/lon

    @staticmethod
    def latlon_to_thetaphi(lat, lon):
        # convert lat to 0 to pi
        lat = (lat+90.)*(numpy.pi/180.)

        # convert lon to 0 to 2pi
        lon = copy.deepcopy(lon)
        if (type(lon) == int) or (type(lon) == float):
            lon = lon + 360. if lon < 0 else lon
        else:
            print type(lon)
            lon[lon < 0] = lon[lon < 0] + 360.
        lon = lon * (numpy.pi/180.)

        return lat, lon
    
    def nc_to_frame(self, path, use_masked = False): # xxx make static method with all inputs?
        ''' converts NetCDF dataset into pandas DataFrame, removing missing 
        values and regions outside lat/lon range + iterpolation buffer. keeps
        only date, time, lat, lon, and inputs'''
        ds = ncdf.Dataset(path) # NetCDF dataset
        n_rows = ds.variables[ds.variables.keys()[-1]].shape[1]
        df = pandas.DataFrame(index = numpy.arange(n_rows))
        meta = pandas.DataFrame()
        
        if self.inputs is None:
            keep = ds.variables.keys() # if no inputs given, keep everything
        else:
            keep = self.inputs + ['img_date', 'img_time', 'lat_cell', 'lon_cell']
        
        # print 'NetCDF keys: ', ds.variables.keys()
        for head, var in ds.variables.items():
            # only keep subset of data we care about
            if head in keep:
                

                data, mask = self._data_from_var(var)
                if len(data) == 1:
                    data = data.repeat(n_rows)
                else: 
                    #print len(data)
                    #print data.shape
                    #print n_rows
                    assert len(data) == n_rows 
                
                # set missing values to nan
                if 'SCALED_MISSING' in var.ncattrs():
                    missing_val = var.SCALED_MISSING
                    data[data == missing_val] = numpy.nan
                
                if not use_masked:
                    # set masked values to nan
                    if mask is not None: 
                        data[mask] = numpy.nan
                
                #print var.__dict__
                df[head] = data # add column to data frame
                
                # collect meta data for variable
                row = pandas.DataFrame([var.__dict__])
                meta = pandas.concat([meta, row], ignore_index=True)
            
        # remove data outside region of interest
        if self.lat_range is not None:
            assert (self.lon_range is not None) & (self.interp_buffer is not None)
            lat = df['lat_cell'] 
            lon = df['lon_cell']
            df = df[(lat >= (self.lat_range[0] - self.interp_buffer[0])) 
                  & (lat <= (self.lat_range[1] + self.interp_buffer[0])) 
                  & (lon >= (self.lon_range[0] - self.interp_buffer[1]))
                  & (lon <= (self.lon_range[1] + self.interp_buffer[1]))]
        
        meta.index = pandas.Index([k for k in ds.variables.keys() if k in keep])

        return df, meta

    def _data_from_var(self, var):
        ''' unpacks a netCDF Variable and returns the array of data and mask if
        the variable is a masked array, otherwise None'''
        arr = var[:].astype(float)
        if type(arr) is numpy.ma.core.MaskedArray:
            return arr.data.T, arr.mask.T
        elif type(arr) is numpy.ndarray:
            return arr.T, None
    
    # XXX doesn't need to be different for different samples; one file = one goes data
    def convert_to_datetime(self, img_date, img_time):
        ''' convert from two arrays of date and time in GOES NetCDF format to 
        pandas datetime '''
        n_pts = len(img_date)
        date = map(lambda x: str(int(x)), img_date)
        year = map(lambda x: 1900 + int(x[:3]), date)
        jday = map(lambda x: int(x[3:]), date) # number of days since new year
        dts = [pandas.datetime.fromordinal(
                        pandas.datetime(year[i], 1, 1).toordinal() + jday[i]) 
                        for i in xrange(n_pts)]
        # add hour to datetime
        hour = (img_time / 10000).astype(int)
        return [pandas.datetime(
                            dts[i].year, 
                            dts[i].month, 
                            dts[i].day, 
                            hour[i]) for i in xrange(n_pts)]


# swd, frac_ice/water/total, tau, olr, 
@plac.annotations(path = 'path to netCDF (.nc) file')
def main(
    path = 'data/satellite/download.class.ngdc.noaa.gov/download/123483484/001/gsipL3_g13_GENHEM_2013122_1', 
    sample_shape = (10,10,2), # (n_lat, n_lon, n_time)
    lat_range = (34., 38.),
    lon_range = (-100., -96.), #oklahoma
    dlat = 0.1,
    dlon = 0.1,
    interp_buffer = (2,2),
    dens_thresh = 0.5,
    sample_stride = (5, 5),
    delta_time = 1., # in hours
    inputs = [
            #'ch2','ch2_cld','ch2_std',
            #'ch9',
            #'ch14','ch14_std','ch14clr', 
            #'frac_snow',
            #'lst', 'trad',
            #'iwp', 'lwp',
            #'cld_temp','cld_press',
            'tau',
            'frac_total_cld', # 'frac_ice_cld', 'frac_water_cld',
            #'ch2_ccr', 'ch2_ccr_std',
            #'olr', 
            'swd_sfc', #'swd_toa', 'swd_sfc_clr', 
            #'swu_sfc', 'swu_toa', 'swu_toa_clr', 'swu_sfc_clr', 
            #'vis_down_sfc', 
            #'cld_type',
            #'ozone',
            #'tpw',
            #'lwdsfc',
            #'lwusfc'
            ]):
    
    paths = glob.glob(path + '*.nc')
    if lat_range is None:
        rad_lat_min = rad_lon_min = None
        rad_lat_max = rad_lon_max = None
    else:
        rad_lat_min, rad_lon_min = GoesData.latlon_to_thetaphi(lat_range[0], lon_range[0])
        rad_lat_max, rad_lon_max = GoesData.latlon_to_thetaphi(lat_range[1], lon_range[1])

    lat_diff = (rad_lat_max - rad_lat_min)
    lon_diff = (rad_lon_max - rad_lon_min)

    dlat = dlat * numpy.pi / 180. # convert to radians
    dlon = dlon * numpy.pi / 180. 
    
    # check that sample window is smaller than observation window
    assert (dlat * sample_shape[0]) < lat_diff
    assert (dlon * sample_shape[1]) < lon_diff
    n_lat_cells = numpy.ceil(lat_diff / dlat).astype(int)
    n_lon_cells = numpy.ceil(lon_diff / dlon).astype(int)

    lat_grid, lon_grid = numpy.mgrid[
        rad_lat_min : rad_lat_max : n_lat_cells*1j,
        rad_lon_min : rad_lon_max : n_lon_cells*1j]
    
    
    samples = {}
    for j, p in enumerate(paths):

        print 'file number: ', j
        
        
        # data from the given netcdf file
        gd = GoesData(p, inputs, lat_range, lon_range, interp_buffer)

        interp_data = numpy.ones((n_lat_cells, n_lon_cells, len(inputs))) * numpy.nan
        # perform interpolation for each input
        for i, inp in enumerate(inputs):
            not_missing = gd.frame.dropna(how='any', subset=[inp])
            print 'length of clean points for %s: ' % inp, len(not_missing)
            
            if len(not_missing) > 0:
                lat = not_missing['lat_rad'].values
                lon = not_missing['lon_rad'].values
                val = not_missing[inp].values
                    
                print 'interpolating to grid'
                interp_data[:,:,i] = interpolate.griddata((lat,lon), val, (lat_grid, lon_grid), method = 'cubic')

        # drop grid cells missing any of the inputs
        gd.frame['lat_ind'] = numpy.floor((gd.frame['lat_rad'] - rad_lat_min) / dlat).astype(int)
        gd.frame['lon_ind'] = numpy.floor((gd.frame['lon_rad'] - rad_lon_min) / dlon).astype(int)
        grp = gd.frame.groupby(['lat_ind', 'lon_ind'])
        grid = grp.aggregate(numpy.mean)
        grid.reset_index(inplace = True)
        grid = grid.dropna(how = 'any', subset = inputs)
        
        # for each window position given stride length
        for x, y in it.product(numpy.arange(0, n_lat_cells - sample_shape[0], sample_stride[0]), 
                               numpy.arange(0, n_lon_cells - sample_shape[1], sample_stride[1])):
            print 'window position: ', x, y

            lat_ind = grid['lat_ind']
            lon_ind = grid['lon_ind']
            present = grid[ (lat_ind >= x) & (lat_ind < (x + sample_shape[0])) &
                            (lon_ind >= y) & (lon_ind < (y + sample_shape[1])) ]
            # if density of observed data is high enough
            print len(present)
            print (dens_thresh * sample_shape[0]*sample_shape[1])
            if len(present) > (dens_thresh * sample_shape[0]*sample_shape[1]):
                print 'storing sample'
                # store this window as a sample by timestep and position
                samples[(gd.timestamp, (x,y))] = interp_data[x:x+sample_shape[0], 
                                                             y:y+sample_shape[1],
                                                              :]
    # convert samples dict to DataFrame
    samp_keys = samples.keys()
    datetimes = [k[0] for k in samp_keys]
    positions = [k[1] for k in samp_keys]
    arrays = pandas.Series(samples.values())
    sample_df = pandas.DataFrame({'datetime':datetimes, 
                                'position':positions, 
                                'array' : arrays})
    
    datetimes = sample_df['datetime']
    positions = sample_df['position']
    d_time = pandas.DateOffset(hours = delta_time)

    # for all one time step samples find neighboring times with same location      
    for i in sample_df.index:
        
        row = sample_df.iloc[i]
        dt = row['datetime']
        pos = row['position']
        
        # XXX extend this to more than one time step
        next_row = sample_df[(datetimes == (dt+d_time)) & (positions == pos)]
        if len(next_row) > 0:
            print 'sample collected'
            print row
            print next_row.iloc[0]  

    
#def parse_imager(
    #path ='data/satellite/download.class.ngdc.noaa.gov/download/123483494/001/goes13.2013.121',
    #sample_shape = (50,50,2), # (n_lat, n_lon, n_time)
    #lat_range = None, #(34., 37.), #(-50, 50), # , #
    #lon_range = None, #(-100., -95.), #(170, 280), # (95, 100), #oklahoma
    #interp_buffer = (2,2),
    #use_masked = True,
    #inputs = [
            #u'version', 
            #u'sensorID', 
            #u'imageDate', 
            #u'imageTime', 
            #u'startLine', 
            #u'startElem', 
            #u'time', 
            #u'dataWidth', 
            #u'lineRes', 
            #u'elemRes', 
            #u'prefixSize', 
            #u'crDate', 
            #u'crTime', 
            #u'bands', 
            ##u'auditTrail', 
            #u'data', 
            #u'lat', 
            #u'lon']):



if __name__ == '__main__':
    plac.call(main)
