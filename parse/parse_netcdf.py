import os
import plac
import glob
import numpy
import pandas
import functools
import itertools as it
import multiprocessing as mp
import cPickle as pickle
import netCDF4 as ncdf
import scipy.interpolate

class GoesData(object):

    ''' represents a single NetCDF (.nc) file with utilities to rescale the data
    into the native units and parse into a pandas dataframe '''
    
    def __init__(self, path, inputs, lat_range = None, lon_range = None, 
                interp_buffer = None, use_masked = True, gzip = False):
        
        self.inputs = inputs
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.interp_buffer = interp_buffer
        self.gzip = gzip
        
        # read netcdf file into pandas data frame
        self.frame, self.meta = self.nc_to_frame(path, use_masked = use_masked) # removes all but lat, lon, inputs, and datetime
    
        if self.frame is not None:
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
    
        if '.gz' == path[-3:]: 
            os.system("gunzip %s" % path)
            path = path[:-3]
        else: assert ('.nc' == path[-3:])
        
        try:
            ds = ncdf.Dataset(path) # NetCDF dataset
            if self.gzip:
                os.system("gzip --fast %s" % path)
        except RuntimeError as e:
            print 'corrupted file, deleting: ', path
            print e
            os.system("rm %s" % path)
            return None, None

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
             n_lat_cells, 
             n_lon_cells,
             lat_grid,
             lon_grid,
             rad_lat_min,
             rad_lon_min,
             dlat,
             dlon,
             sample_stride,
             dens_thresh,
             window_shape,
             gzip
             ):
    ''' 
    parses a netCDF file into a dictionary of samples restricting to lat and lon
    range and requiring at least dens_thresh observation density. 

     path : path to netCDF file, can be *.nc or *.nc.gz
     inputs : list of input channels to be used
     lat_range : tuple (latitude min, latitude_max) 
     lon_range : tuple (longitude min, longitude_max) 
     interp_buffer : tuple of pixels to pad around used region for interpolation 
     n_lat_cells : number of latitude pixels/cells in entire region of interest
     n_lon_cells : number of longitude pixels/cells in entire region of interest
     lat_grid : list of latitude positions for interpolation grid
     lon_grid : list of longitude positions for interpolation grid
     rad_lat_min : min latitude in radians (see latlon_to_thetaphi above)
     rad_lon_min : min longitude in radians (see latlon_to_thetaphi above) 
     dlat : delta latitude in radians for each grid pixel/cell
     dlon : delta longitude in radians for each grid pixel/cell
     sample_stride : tuple of number of pixels to move when scanning for samples 
     dens_thresh : observation density necessary for 'valid' samples 
     window_shape : tuple of window size (width, height)
     gzip : boolean, zip files after reading them? (can still read *.nz.gz if false)
    '''

    samples = {}
    
    gd = GoesData(path, inputs, lat_range, lon_range, interp_buffer, gzip = gzip)
    
    if gd.frame is None: return None

    n_wind_cells = window_shape[0]*window_shape[1]
    n_channels = len(inputs)

    interp_data = numpy.zeros((n_channels, n_lat_cells, n_lon_cells)) * numpy.nan
    # perform interpolation for each input
    for i, inp in enumerate(inputs):
        not_missing = gd.frame.dropna(how='any', subset=[inp])
        #print 'length of clean points for %s: ' % inp, len(not_missing)
        
        if len(not_missing) > 4: # min needed for griddata interpolation
            lat = not_missing['lat_rad'].values
            lon = not_missing['lon_rad'].values
            val = not_missing[inp].values
            try:
                result = scipy.interpolate.griddata((lat,lon), val, 
                                        (lat_grid,lon_grid), method = 'cubic')

                #pyplot.imshow(result, cmap = 'jet') # XXX move to indep func?
                #pyplot.savefig('interpolated.%i.%f.png' % (i, numpy.random.random()))
                #print 'result shape: ', result.shape
                interp_data[i,:,:] = result
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
        # XXX is this the best way to determine observation density?
        # if density of observed data is high enough and there are no nans in the interpolated data
        if (len(present) > (dens_thresh * n_wind_cells)) & (not numpy.isnan(interp_window).any()):
            # store this window as a sample by timestep and position
            samples[(gd.timestamp, (x,y))] = interp_window
    
    return samples

def split_loc_samples(data_frame, n_frames, n_channels, delta_time, window_shape):

    ''' find sequences of n_frames + 1 with delta_time (in hours) between each 
    frame. The last frame's center pixel is used as the target value'''

    data_frame.reset_index(inplace = True)
    
    d_time = pandas.DateOffset(hours = delta_time)
    n_wind_cells = window_shape[0]*window_shape[1]
    center_ind = (numpy.ceil(window_shape[0]/2.), numpy.ceil(window_shape[1]/2.))
    datetimes = data_frame['datetime']

    windows = None; targets = None; timestamps = None
    for i in data_frame.index:
 
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

def pickle_dat(i, dataset, out_path, gzip = False):

    ''' pickles the given dataset dictionary with file number i in the out path
    directory. Additionally gzips if gzip is True.
    '''
    
    from solar.util import openz

    delta_time = dataset['dt']
    n_frames = dataset['n-frames']
    n_channels = dataset['n-channels']
    _, nf, nc, n_wind_cells = dataset['windows'].shape
    n_samples = dataset['n-samples'] 
    window_size = numpy.sqrt(n_wind_cells).astype(int)
    assert nf == n_frames
    assert nc == n_channels
    assert window_size**2 == n_wind_cells
    sample_stride = dataset['sample-stride']
    dens_thresh = dataset['dens-thresh']
    
    path_name = out_path + 'goes-insolation.dt-%0.1f.nf-%i.nc-%i.ws-%i.str-%i.dens-%0.1f.nsamp-%i.%i.pickle'
    if gzip:
        path_name = path_name + '.gz'
    with openz(path_name % (delta_time, n_frames, n_channels, window_size, 
               sample_stride[0], dens_thresh, n_samples, i), 'wb') as pfile:
        pickle.dump(dataset, pfile)


# plac annotations are of the form (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
in_path = ('path and option prefix for netCDF (.nc) files, can use regular expressions (as with ls)', 'option', None, str),
out_path = ('directory where processed files will be placed', 'option', None, str),
window_size = ('size of sample grid window, assumed square', 'option', None, int),
n_frames = ('number of past timesteps into the past used for prediction', 'option', None, int),
delta_time = ('difference in time between samples in hours', 'option', None, float),    
lat_range = ('2-tuple of min and max latitudes to use for samples in degrees, ex: OK is (34., 38.)', 'option', None, None),
lon_range = ('2-tuple of min and max longitudes to use for samples in degrees, ex: OK is (-100., -96.)', 'option', None, None),
dlat = ('delta latitude for interpolated grid', 'option', None, float),
dlon = ('delta longitude for interpolated grid', 'option', None, float),
interp_buffer = ('2-tuple in degrees of buffer around lat/lon_range to include for interpolation', 'option', None, None),
dens_thresh = ('fraction of full observation density that must be present for sample to be considered valid', 'option', None, float),
sample_stride = ('grid cells (pixels) to move over/down when scanning to collect samples', 'option', None, int),
normalized = ('boolean switch for setting data mean to 0 and covariance to 1' , 'option', None, bool),
nper_file = ('number of samples per file', 'option', None, int),
gzip = ('should we zip the processed files while pickling, trades speed for space', 'option', None, bool),
inputs = ('list of variable (short) names to be used as input channels in samples', 'option', None, None)
)
def main(
    in_path = 'data/satellite/raw/', 
    out_path = 'data/satellite/processed/', 
    window_size = 9,
    n_frames = 1, 
    delta_time = 1., 
    lat_range = (34., 38.), 
    lon_range = (-100., -96.),
    dlat = 0.1,
    dlon = 0.1,
    interp_buffer = (2,2),
    dens_thresh = 0.6,
    sample_stride = 1,
    normalized = True,
    nper_file = 50000,
    gzip = False,
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
    
    zipped_paths = glob.glob(in_path + '*.nc.gz')
    paths = glob.glob(in_path + '*.nc')
    paths.extend(zipped_paths)
    paths = sorted(paths)

    assert len(paths) > 0

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
    
    sample_stride = (sample_stride, sample_stride)
    window_shape = (window_size, window_size)
    # check that sample window is smaller than observation window
    assert (dlat * window_shape[0]) < lat_diff
    assert (dlon * window_shape[1]) < lon_diff

    n_lat_cells = numpy.ceil(lat_diff / dlat).astype(int)
    n_lon_cells = numpy.ceil(lon_diff / dlon).astype(int)
    n_channels = len(inputs)

    lat_grid, lon_grid = numpy.mgrid[
        rad_lat_min : rad_lat_max : n_lat_cells*1j,
        rad_lon_min : rad_lon_max : n_lon_cells*1j]
    
    print 'processing files from ', in_path
    print 'window shape: ', window_shape
    print 'delta time (hours): ', delta_time
    print 'number of time lags: ', n_frames
    print 'number of channels: ', n_channels

    part_parse_nc = functools.partial(parse_nc, 
                     inputs = inputs, 
                     lat_range = lat_range, 
                     lon_range = lon_range, 
                     interp_buffer = interp_buffer, 
                     n_lat_cells = n_lat_cells, 
                     n_lon_cells = n_lon_cells,
                     lat_grid = lat_grid,
                     lon_grid = lon_grid,
                     rad_lat_min = rad_lat_min,
                     rad_lon_min = rad_lon_min,
                     dlat = dlat,
                     dlon = dlon,
                     sample_stride = sample_stride,
                     dens_thresh = dens_thresh,
                     window_shape = window_shape,
                     gzip = gzip)

    samples = {}
    #os.system("taskset -p 0xffffffffffffffff %d" % os.getpid()) # sometimes numpy restricts process to using 1 cpu, undoes this
    pool = mp.Pool(int(mp.cpu_count()*0.75))
    dicts = pool.map_async(part_parse_nc, paths).get()
    map(samples.update, dicts)

    assert len(samples) > 0

    pool.close()
    pool.join()

    print 'data read from netcdf paths'

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
            if winds is not None:
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
    
    pool = mp.Pool(mp.cpu_count())
    for name, group in sample_gb:
        #print 'group: ', name
        pool.apply_async(split_loc_samples, args = [group, n_frames, n_channels, delta_time, window_shape], callback = sample_set.append_sample)
        #sample_set.append_sample(apply(split_loc_samples, [group, n_frames, n_channels, delta_time, window_shape]))

    pool.close()
    pool.join()
    
    if sample_set.windows is None: assert False

    assert sample_set.windows is not None

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
    n_samples = sample_set.windows.shape[0]
    n_files = numpy.ceil(n_samples / float(nper_file)).astype(int)

    #os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())
    pool = mp.Pool(min(mp.cpu_count(), n_files))
    
    print 'number of samples collected: ', n_samples
    print 'saving to file'
    for i in xrange(n_files):
        print 'file number: ', i
        dataset = {
                  'windows': sample_set.windows[i*nper_file:(i+1)*nper_file],
                  'targets': sample_set.targets[i*nper_file:(i+1)*nper_file],
                  'timestamps': sample_set.timestamps[i*nper_file:(i+1)*nper_file],
                  'input-names': inputs, 
                  'norm-stats': stats,
                  'n-samples': n_samples,
                  'n-frames': n_frames,
                  'n-channels':n_channels,
                  'sample-stride':sample_stride,
                  'dens-thresh':dens_thresh,
                  'lat-range':lat_range,
                  'lon-range':lon_range,
                  'dlat':dlat,
                  'dlon':dlon,
                  'dt':delta_time,
                  'window-shape': window_shape,
                  'normalized':normalized,
                  'data-path':in_path
                  }
        print 'number of samples in file: ', dataset['windows'].shape[0]
        assert dataset['windows'].shape[0] <= nper_file
        pool.apply_async(pickle_dat, args = [i, dataset, out_path, gzip])
        
    pool.close()
    pool.join()
    print 'done'

if __name__ == '__main__':
    plac.call(main)
