import plac
import glob
import copy
import numpy
import pandas
import functools
import itertools as it
import multiprocessing as mp
import cPickle as pickle
import netCDF4 as ncdf
import matplotlib.pyplot as pyplot
from scipy import interpolate
from solar.util import openz

# XXX whats the difference between missing and mask?
# XXX store non-spatial parameters separately?

class GoesData(object):

    ''' represents a single NetCDF (.nc) file with utilities to rescale the data
    into the native units and parse into a pandas dataframe'''
    
    def __init__(self, path, inputs, lat_range = None, lon_range = None, 
                interp_buffer = None, use_masked = True):
        
        self.inputs = inputs
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.interp_buffer = interp_buffer
        
        # read netcdf file into pandas data frame
        self.frame, self.meta = self.nc_to_frame(path, use_masked = use_masked) # removes all but lat, lon, inputs, and datetime

        self.timestamp = self.convert_to_datetime(
                                            self.frame['img_date'].values[0],
                                            self.frame['img_time'].values[0])
        self.rescale()
        

    def rescale(self):
        '''rescale by units in metadata'''

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

    @staticmethod
    def thetaphi_to_latlon(theta, phi):
        theta = theta * (180./numpy.pi) - 90.
        phi = phi * (180./numpy.pi) - 180.
        return theta, phi

    @staticmethod
    def latlon_to_thetaphi(lat, lon):
        lat = (lat + 90. ) * (numpy.pi/180.) # convert lat to 0 to pi
        lon = (lon + 180.) * (numpy.pi/180.) # convert lon to 0 to 2pi
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
                    assert len(data) == n_rows 
                
                # set missing values to nan
                if 'SCALED_MISSING' in var.ncattrs():
                    missing_val = var.SCALED_MISSING
                    data[data == missing_val] = numpy.nan
                
                if not use_masked:
                    # set masked values to nan
                    if mask is not None: 
                        data[mask] = numpy.nan
                
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
    
    def convert_to_datetime(self, img_date, img_time):
        ''' convert from date and time in GOES NetCDF format to pandas datetime 
        '''
        date = str(int(img_date))
        year = 1900 + int(date[:3])
        jday = int(date[3:])
        datetime = pandas.datetime.fromordinal(
                                pandas.datetime(year,1,1).toordinal() + jday)
        # add hour to datetime
        hour = (img_time / 10000).astype(int)
        return pandas.datetime(
                            datetime.year, 
                            datetime.month, 
                            datetime.day, 
                            hour) 

def parse_nc(
             path, 
             inputs, 
             lat_range, 
             lon_range, 
             interp_buffer, 
             n_channels, 
             n_lat_cells, 
             n_lon_cells,
             n_wind_cells,
             lat_grid,
             lon_grid,
             rad_lat_min,
             rad_lon_min,
             dlat,
             dlon,
             sample_stride,
             dens_thresh,
             window_shape
             ):

    #print 'parsing', path
    samples = {}
    
    gd = GoesData(path, inputs, lat_range, lon_range, interp_buffer)
    interp_data = numpy.ones((n_channels, n_lat_cells, n_lon_cells)) * numpy.nan
    # perform interpolation for each input
    for i, inp in enumerate(inputs):
        not_missing = gd.frame.dropna(how='any', subset=[inp])
        #print 'length of clean points for %s: ' % inp, len(not_missing)
        
        if len(not_missing) > 4:
            lat = not_missing['lat_rad'].values
            lon = not_missing['lon_rad'].values
            val = not_missing[inp].values
            try:
                interp_data[i,:,:] = interpolate.griddata((lat,lon), val, (lat_grid, lon_grid), method = 'cubic')
            except Exception as e:
                print e

    # drop grid cells missing any of the inputs
    gd.frame['lat_ind'] = numpy.floor((gd.frame['lat_rad'] - rad_lat_min) / dlat).astype(int)
    gd.frame['lon_ind'] = numpy.floor((gd.frame['lon_rad'] - rad_lon_min) / dlon).astype(int)
    grp = gd.frame.groupby(['lat_ind', 'lon_ind'])
    grid = grp.aggregate(numpy.mean)
    grid.reset_index(inplace = True)
    grid = grid.dropna(how = 'any', subset = inputs)
    
    # for each window position given stride length
    for x, y in it.product(numpy.arange(0, n_lat_cells - window_shape[0], sample_stride[0]), 
                           numpy.arange(0, n_lon_cells - window_shape[1], sample_stride[1])):

        lat_ind = grid['lat_ind']
        lon_ind = grid['lon_ind']
        present = grid[ (lat_ind >= x) & (lat_ind < (x + window_shape[0])) &
                        (lon_ind >= y) & (lon_ind < (y + window_shape[1])) ]

        interp_window = interp_data[:,x:x+window_shape[0], 
                                      y:y+window_shape[1]]
        # XXX best way to determine observation density?
        # if density of observed data is high enough and there are no nans in the interpolated data
        if (len(present) > (dens_thresh * n_wind_cells)) & (not numpy.isnan(interp_window).any()):
            # store this window as a sample by times    tep and position
            samples[(gd.timestamp, (x,y))] = interp_window

    return samples

def split_loc_samples(data_frame, n_frames, n_channels, delta_time, window_shape):
    ''' find sequences of n_frames + 1 with delta_time (in hours) between each 
    frame. The last frame's center pixel is used as the target value'''
    
    #print 'parsing location frame'

    data_frame.reset_index(inplace = True)
    
    d_time = pandas.DateOffset(hours = delta_time)
    n_wind_cells = window_shape[0]*window_shape[1]
    center_ind = (numpy.ceil(window_shape[0]/2.), numpy.ceil(window_shape[1]/2.))
    datetimes = data_frame['datetime']

    windows = None; targets = None; timestamps = None
    for i in data_frame.index:

        #print 'row %i of %i' % (i, len(sample_df.index))
        
        row = data_frame.iloc[i]
        time = row['datetime']
        
        mask = numpy.zeros(len(datetimes), dtype=bool)
        for t in xrange(n_frames):
            mask = mask | (datetimes == (time+(t+1)*d_time))
            
        next_rows = data_frame[mask]
        if len(next_rows) == (n_frames): # if there are n_frames valid frames
            next_rows.sort('datetime')
            winds = numpy.empty((n_frames, n_channels, n_wind_cells))
            winds[0] = numpy.reshape(row['array'], (n_channels, n_wind_cells))

            # flatten all but the last frame, the last used for target
            for w in xrange(n_frames-1): 
                winds[w+1] = numpy.reshape(next_rows.iloc[w]['array'], (n_channels, n_wind_cells)) 
            
            times = numpy.append([time], next_rows['datetime'])

            if windows is None:
                windows = winds[None,:,:,:]
                targets = next_rows.iloc[-1]['array'][None,:,center_ind[0],center_ind[1]]
                timestamps = times[None,:]
            else:
                windows = numpy.append(windows, winds[None,:,:,:], axis = 0)
                targets = numpy.append(targets, next_rows.iloc[-1]['array'][None,:,center_ind[0],center_ind[1]], axis = 0)
                timestamps = numpy.append(timestamps, times[None,:], axis = 0)
                
                assert windows.shape[0] == targets.shape[0] == timestamps.shape[0] 
    
    return windows, targets, timestamps


# swd, frac_ice/water/total, tau, olr, 
#(help, kind, abbrev, type, choices, metavar)
@plac.annotations(
path = ('path to netCDF (.nc) file', 'option', None, str),
window_size = ('2-tuple shape of sample grid window', 'option', None, int),
n_frames = ('number of past timesteps into the past used for prediction', 'option', None, int),
delta_time = ('difference in time between samples in hours', 'option', None, float),    
lat_range = ('2-tuple of min and max latitudes to use for samples in degrees', 'option', None, None),
lon_range = ('2-tuple of min and max longitudes to use for samples in degrees', 'option', None, None),
dlat = ('delta latitude for interpolated grid', 'option', None, float),
dlon = ('delta longitude for interpolated grid', 'option', None, float),
interp_buffer = ('2-tuple in degrees of buffer around lat/lon_range to include for interpolation', 'option', None, None),
dens_thresh = ('fraction of full observation density that must be present for sample to be considered valid', 'option', None, float),
sample_stride = ('grid cells (pixels) to move over/down when scanning to collect samples', 'option', None, None),
normalized = ('boolean switch for setting data mean to 0 and covariance to 1' , 'option', None, bool),
inputs = ('list of variable (short) names to be used as input channels in samples', 'option', None, None)
)
def main(
    path = 'data/satellite/raw/', # gsipL3_g13_GENHEM_20131',
    window_size = 9, # (n_lat, n_lon)
    n_frames = 1, # number of frames into the past used for prediction 
    delta_time = 1., # in hours
    lat_range = (34., 38.),
    lon_range = (-100., -96.), #oklahoma
    dlat = 0.1,
    dlon = 0.1,
    interp_buffer = (2,2),
    dens_thresh = 0.6,
    sample_stride = (1, 1),
    normalized = True,
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
    ''' 
    A script for parsing a collection of netcdf files into a dictionary of 
    input grids and target values for the given input channels
    '''
    
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
    
    window_shape = (window_size, window_size)
    # check that sample window is smaller than observation window
    assert (dlat * window_shape[0]) < lat_diff
    assert (dlon * window_shape[1]) < lon_diff

    n_lat_cells = numpy.ceil(lat_diff / dlat).astype(int)
    n_lon_cells = numpy.ceil(lon_diff / dlon).astype(int)
    n_wind_cells = window_shape[0]*window_shape[1]
    n_channels = len(inputs)

    lat_grid, lon_grid = numpy.mgrid[
        rad_lat_min : rad_lat_max : n_lat_cells*1j,
        rad_lon_min : rad_lon_max : n_lon_cells*1j]
    
    print 'processing files from ', path
    print 'window shape: ', window_shape
    print 'delta time (hours): ', delta_time
    print 'number of time lags: ', n_frames
    print 'number of channels: ', n_channels


    part_parse_nc = functools.partial(parse_nc, # xxx make number of params smaller?
                     inputs = inputs, 
                     lat_range = lat_range, 
                     lon_range = lon_range, 
                     interp_buffer = interp_buffer, 
                     n_channels = n_channels, 
                     n_lat_cells = n_lat_cells, 
                     n_lon_cells = n_lon_cells,
                     n_wind_cells = n_wind_cells,
                     lat_grid = lat_grid,
                     lon_grid = lon_grid,
                     rad_lat_min = rad_lat_min,
                     rad_lon_min = rad_lon_min,
                     dlat = dlat,
                     dlon = dlon,
                     sample_stride = sample_stride,
                     dens_thresh = dens_thresh,
                     window_shape = window_shape)
    
    pool = mp.Pool(mp.cpu_count())
    samples = {}
    dicts = pool.map_async(part_parse_nc, paths).get()
    map(samples.update, dicts)
    assert len(samples) > 0

    # convert samples dict to DataFrame
    samp_keys = samples.keys()
    datetimes = [k[0] for k in samp_keys]
    positions = [k[1] for k in samp_keys]
    arrays = pandas.Series(samples.values())
    sample_df = pandas.DataFrame({'datetime':datetimes, 
                                'position':positions, 
                                'array' : arrays})
    
    print 'finding neighboring frames of time-length %i with delta time %0.1f hours' % (n_frames + 1, float(delta_time))

    sample_gb = sample_df.groupby(['position']) 

    class SampleSet(object):
        
        def __init__(self):
            
            self.windows = None
            self.targets = None
            self.timestamps = None

        def append_sample(self, sample_tuple):
            winds, targs, times = sample_tuple
            assert winds.ndim == 4 # (n_samples, n_frames, n_channels, n_wind_cells) 
            assert targs.ndim == 2 # (n_samples, n_channels)
            assert times.ndim == 2 # (n_samples, n_frames) 
            
            if winds.shape[0] > 0:
                if self.windows is None:
                    self.windows = winds
                    self.targets = targs
                    self.timestamps = times
                else:
                    self.windows = numpy.append(self.windows, winds, axis = 0)
                    self.targets = numpy.append(self.targets, targs, axis = 0)
                    self.timestamps = numpy.append(self.timestamps, times, axis = 0)

    sample_set = SampleSet()
    
    for name, group in sample_gb:
        pool.apply_async(split_loc_samples, 
                args = [group, n_frames, n_channels, delta_time, window_shape], 
                callback = sample_set.append_sample)
    pool.close() 
    pool.join()

    print 'sample input shape: ', sample_set.windows.shape

    stats = numpy.zeros((n_channels,2)); stats[:,1] = 1.
    if normalized: 
        print 'normalizing the data'
        # normalize the data (only using windows, ignoring targets currently)
        for i in xrange(n_channels):
            mn = numpy.mean(sample_set.windows[:,:,i,:])
            std = numpy.std(sample_set.windows[:,:,i,:])
            stats[i,:] = [mn, std]
            
            sample_set.windows[:,:,i,:] = sample_set.windows[:,:,i,:] - mn
            sample_set.windows[:,:,i,:] = sample_set.windows[:,:,i,:] / std

            sample_set.targets[:,i] = sample_set.targets[:,i] - mn
            sample_set.targets[:,i] = sample_set.targets[:,i] / std
            
            print inputs[i], ': '
            print 'window mean: ', mn
            print 'window std: ', std
            print 'target mean after normalization: ', numpy.mean(sample_set.targets[:,i])
            print 'target std after normalization: ', numpy.std(sample_set.targets[:,i])
    
    # windows is shape (n_samples, n_frames, n_channels, n_wind_cells)
    dataset = {
              'windows': sample_set.windows,
              'targets': sample_set.targets,
              'timestamps': sample_set.timestamps,
              'input-names':inputs, 
              'norm-stats': stats,
              'n-frames':n_frames,
              'n-channels':n_channels,
              'sample-stride':sample_stride,
              'dens-thresh':dens_thresh,
              'lat-range':lat_range,
              'lon-range':lon_range,
              'dlat':dlat,
              'dlon':dlon,
              'dt':delta_time,
              'window-shape':window_shape,
              'normalized':normalized,
              'data-path':path
              }
    
    n_samples = sample_set.windows.shape[0]
    print 'number of samples collected: ', n_samples
    print 'saving to file'

    with openz('data/satellite/processed/goes-insolation.dt-%0.1f-nf-%i.nc-%i.ws-%i.str-%i.dens-%0.1f.nsamp-%i.pickle.gz' % 
            (delta_time, n_frames, n_channels, window_shape[0], sample_stride[0], dens_thresh, n_samples), 'wb') as pfile:
        pickle.dump(dataset, pfile)

if __name__ == '__main__':
    plac.call(main)
