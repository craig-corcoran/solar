import plac
import glob
import numpy
import pandas
import itertools as it
import netCDF4 as ncdf
import matplotlib.pyplot as pyplot
from scipy import interpolate
from sklearn.preprocessing import Scaler



# XXX whats the difference between missing and mask?


# XXX weight outliers lower during interpolation

class GoesData(object):
    
    def __init__(self, path_regex, inputs, lat_range = None, lon_range = None, 
                interp_buffer = None):
        
        self.inputs = inputs
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.interp_buffer = interp_buffer

        paths = glob.glob(path_regex)
        
        # read netcdf file
        clean_frames = [None]*len(paths)
        for i,p in enumerate(paths):
            # read into pandas data frame
            df, meta = self.nc_to_frame(p)
            clean_frames[i] = df 

        self.meta = meta # should be the same across files

        print 'meta data: ', meta
        print 'concatenating all files into one frame'

        # combine files into one frame
        self.frame = pandas.concat(clean_frames, ignore_index = True) 

        # add datetime column, remove all but lat, lon, inputs, and datetime
        self.frame['datetime'] = self.convert_to_datetime(
                                            self.frame['img_date'].values,
                                            self.frame['img_time'].values)
        #self.frame = self.frame.sort('datetime')
        
        print 'centering and rescaling data'
        self.rescale()
        

    def rescale(self):
        '''rescale by units in metadata, center and set variance to unity, 
        creates scaler (for applying the same scaling or un-scaling later) '''

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
        self.mean_vec = self.frame[self.inputs].mean()
        self.std_vec = self.frame[self.inputs].std()

        print 'mean: ', numpy.array(self.mean_vec)
        print 'std: ', numpy.array(self.std_vec)

        self.frame[self.inputs] = self.frame[self.inputs].astype(float) - self.mean_vec
        self.frame[self.inputs] = self.frame[self.inputs].astype(float) / self.std_vec

    # XXX unscale
    # convert theta/phi to lat/lon

    @staticmethod
    def latlon_to_thetaphi(lat, lon):
        # convert lat to 0 to pi
        lat = (lat+90.)*(numpy.pi/180.)

        # convert lon to 0 to 2pi
        lon[lon < 0] = lon[lon < 0] + 360.
        lon = lon * (numpy.pi/180.)

        return lat, lon
    
    def nc_to_frame(self, path): # xxx make static method with all inputs?
        ''' converts NetCDF dataset into pandas DataFrame, removing missing 
        values and regions outside lat/lon range + iterpolation buffer'''
        ds = ncdf.Dataset(path) # NetCDF dataset
        n_rows = ds.variables[ds.variables.keys()[-1]].shape[1]
        df = pandas.DataFrame(index = numpy.arange(n_rows))
        meta = pandas.DataFrame()
        
        keep = self.inputs + ['img_date', 'img_time', 'lat_cell', 'lon_cell']
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
        
        meta.index = pandas.Index(keep)

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
    path = 'data/satellite/download.class.ngdc.noaa.gov/download/123411444/001/',
    sample_shape = (50,50,2), # (n_lat, n_lon, n_time)
    lat_range = (-50, 50), #(34, 37),
    lon_range = (170, 280), # (95, 100), # oklahoma
    interp_buffer = (2,4),
    inputs = [
            #'ch2','ch2_cld','ch2_std',
            #'ch9',
            #'ch14','ch14_std','ch14clr', 
            #'frac_snow',
            #'lst', 'trad',
            #'iwp', 'lwp',
            #'cld_temp','cld_press',
            'tau',
            'frac_total_cld', 'frac_ice_cld', 'frac_water_cld',
            #'ch2_ccr', 'ch2_ccr_std',
            'olr', 
            'swd_sfc', 'swd_toa', 'swd_sfc_clr', 
            #'swu_sfc', 'swu_toa', 'swu_toa_clr', 'swu_sfc_clr', 
            #'vis_down_sfc', 
            #'cld_type',
            #'ozone',
            #'tpw',
            #'lwdsfc',
            #'lwusfc'
            ]):
    
    gd = GoesData(path+'*.nc', inputs, lat_range, lon_range, interp_buffer)

    print gd.frame
    print 'new lat/lon bounds: '
    pos_extrema = {'max': numpy.max([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1),
                   'min': numpy.min([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1)}
    print pos_extrema 

    for i, dt in enumerate(gd.frame['datetime']):
        
        time_slice = gd.frame[gd.frame['datetime'] == dt]
        
        for inp in inputs:
            no_missing = time_slice.dropna(how='any', subset=[inp])
            print 'length of clean points for %s: ' % inp, len(no_missing)
            
            lat = no_missing['lat_rad'].values
            lon = no_missing['lon_rad'].values
            val = no_missing[inp].values
            
            print 'clean lat max min: ', numpy.max(lat), numpy.min(lat)
            print 'clean lon max min: ', numpy.max(lon), numpy.min(lon)
          
            spline_built = False
            mult = 0.5
            while not spline_built:
                try:
                    print 'building spline representation' 
                    spline = interpolate.SmoothSphereBivariateSpline(lat, lon, val, 
                                                            s=mult*len(no_missing))
                    spline_built = True
                except ValueError as e:
                    print e
                    mult += 0.2
                    print 'increasing s value to %f times number of points' % mult
            
            frac = 0.2
            dlat = numpy.max(lat) - numpy.min(lat)
            dlon = numpy.max(lon) - numpy.min(lon)
            ilat = numpy.linspace(numpy.min(lat) + frac/2.*dlat, numpy.max(lat) - frac/2.*dlat, sample_shape[0])
            ilon = numpy.linspace(numpy.min(lon) + frac/2.*dlon, numpy.max(lon) - frac/2.*dlon, sample_shape[1])
            
            'interpolating to grid'
            interpol = spline(ilat, ilon)
            
            print interpol
            print interpol.shape

            print 'interpolated data shape: ', interpol.shape
            print 'max, min clean interpolated vals: ', numpy.max(interpol), numpy.min(interpol)

            print 'plotting'
            pyplot.clf()
            ax = pyplot.subplot(121)
            perm = numpy.random.permutation(len(lat))
            n_plot = 10000
            latplot = lat[perm][:n_plot]
            lonplot = lon[perm][:n_plot]
            valplot = val[perm][:n_plot]

            vmin = numpy.min([numpy.min(valplot), numpy.min(interpol)])
            vmax = numpy.max([numpy.max(valplot), numpy.max(interpol)])

            pyplot.scatter(latplot, lonplot, c = valplot, 
                    cmap = 'jet', s=10, linewidths = 0, vmin=vmin, vmax=vmax)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            ilat_grid, ilon_grid = it.izip(*it.product(ilat, ilon))
            pyplot.colorbar()

            ax = pyplot.subplot(122)
            pyplot.scatter(ilat_grid, ilon_grid, c = interpol.flatten(), 
                    cmap = 'jet', s=10, linewidths = 0, vmin=vmin, vmax=vmax)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            pyplot.savefig('sphere_spline_interp-%s-%i.pdf' % (inp, i))

    assert False
    
    # scale data using range_min and range_max
    # normalize (center and set var to 1) -> scikit-learn

    # for each time slice, for each input, iterpolate spatially
    # store interpolated data?
    # sample receptive field on uniform grid, store samples


    
   


    #grouped = frame.groupby(['lat_cell','lon_cell', 'img_date', 'img_time'])
    #mean_data = grouped.mean()
    #print len(mean_data)
    #print len(frame)
    #for inp in inputs:

        #print 'creating spherical spline representation'
        #print numpy.max(frame['lat_cell'].values * (numpy.pi/180.))
        #print numpy.max(frame['lon_cell'].values * (numpy.pi/180.))
        #print numpy.max(frame[inp].values) 

        #print frame['lat_cell'].values.shape
        #print frame['lon_cell'].values.shape
        #print frame[inp].values.shape

        #no_missing = frame.dropna(how='any', subset=[inp])
        ##bispline = interpolate.SmoothSphereBivariateSpline(
                                    ##numpy.linspace(0, numpy.pi, 10),
                                    ##numpy.linspace(0, numpy.pi*2, 10),
                                    ##numpy.random.random(10), 
                                    ##s = 10)

        #bispline = interpolate.SmoothSphereBivariateSpline(
                                #no_missing['lat_cell'].values*(numpy.pi/180.),
                                #no_missing['lon_cell'].values*(numpy.pi/180.),
                                #no_missing[inp].values,
                                #s=len(no_missing)) 
        #print bispline

   
    
    # index lat and lon according to range and dlat, dlon
    x = frame['lat_cell']
    y = frame['lon_cell']
    indx = numpy.vstack([x - pos_extrema['min'][0],y - pos_extrema['min'][1]])
    indx = numpy.floor(indx / dlatlon[:,None]).astype(int)
    frame['lat_index'] = indx[0]
    frame['lon_index'] = indx[1]
    
    #print 'plotting density histogram'
    #pyplot.clf()
    #pyplot.hexbin(x, y)
    #pyplot.savefig('plots/satellite/density.pdf')
    
    # average over lat,lon and time groups
    grouped = frame.groupby(['lat_index','lon_index', 'img_date', 'img_time'])
    mean_data = grouped.mean()
    
    print 'len before dropping na from gridded frame: ', len(mean_data)
    mean_data.dropna(how='any', subset=inputs)
    print 'len after dropping na from gridded frame: ', len(mean_data)
    mean_data = mean_data.reset_index()
    mean_data['datetime'] = convert_to_datetime(mean_data['img_date'].values,
                                                mean_data['img_time'].values)
    mean_data = mean_data.sort('datetime')

    # convert to datetime 
    #float_time = mean_data['datetime'].values.astype(float)
    #assert (float_time == sorted(float_time)).all()
    #dtime = float_time[1:] - float_time[:-1]
    #dtime = dtime[dtime > 0]
    #dtime = numpy.min(dtime)

    max_pos_indx = numpy.max(indx, axis = 1)
    size_x = max_pos_indx[0]+1
    size_y = max_pos_indx[1]+1
    lat_grid, lon_grid = numpy.mgrid[0:size_x, 0:size_y]

    # for each time slice, interpolate spatially
    for i, time in enumerate(sorted(set(mean_data['datetime']))):
        
        print 'time slice month, day, hr, min: ', time.month, time.day, time.hour, time.minute

        time_slice = mean_data[mean_data['datetime'] == time]

        points = map(lambda x: x[:,None], [time_slice['lat_index'].values, 
                                            time_slice['lon_index'].values])
        points = numpy.hstack(points)
        n_samp = points.shape[0]
        print 'number of grid cells: ', n_samp
        print 'grid size: ', lat_grid.shape
        
        interp_data = numpy.empty((size_x, size_y, len(inputs)), dtype = 'float')
        interp_data[:] = numpy.nan
        for j, inp in enumerate(inputs):

            print 'input: ', inp
            # spatially interpolate over the input channel
            
            spline_rep = interpolate.bisplrep(points[0,:], points[1,:], 
                                time_slice[inp].values.astype(float), 
                                s=n_samp + numpy.sqrt(2*n_samp)) # upper bound on s given in spline docs
            z_grid = interpolate.bisplev(lat_grid, lon_grid, spline_rep)
            # z_grid = griddata(points, time_slice[inp].values.astype(float), (lat_grid, lon_grid), method=interp_method)
            interp_data[:,:,j] = z_grid # XXX plot new interpolated data
            
            n_non_null = z_grid.shape[0] * z_grid.shape[1] - numpy.sum(numpy.isnan(z_grid))
            print 'number non null after interpolation: ', n_non_null
            print 'total numer of grid cells: ', size_x * size_y

        occup_grid = numpy.zeros(z_grid.shape, dtype = 'bool')
        occup_grid[points[:,0], points[:,1]] = True
        samples, samp_indxs = split_into_samples(occup_grid, interp_data, 
                                        dens_thresh = 0.5, samp_size = (10, 10)) # xxx set as input

            
            #if n_non_null > 0:
                #pyplot.clf()
                #pyplot.scatter(lat_grid.flatten(), lon_grid.flatten(), c = z_grid.flatten(), s=80, cmap='jet')
                #pyplot.savefig('plots/satellite/%s-%i-postinterp-%s.png' % (inp, i, interp_method))

            #pyplot.savefig(
            #pyplot.savefig('plots/satellite/ch2%i.png' % i)

def split_into_samples(pre_grid, post_grid, dens_thresh = 0.5, samp_size = (10, 10)):
    
    x_size, y_size = post_grid.shape
    x_inds = x_size-samp_size[0]
    y_inds = y_size-samp_size[1]
    print 'initializing filter matrix'
    filters = numpy.zeros((x_inds*y_inds, x_size, y_size), dtype = 'bool')

    print 'filter matrix shape: ', filters.shape
    
    print 'constructing filters' 
    indxs = numpy.array(list(it.product(xrange(x_inds), xrange(y_inds))))
    for k, (i, j) in enumerate(indxs):
        filters[k, i:i+samp_size[0], j:j+samp_size[1]] = True
    
    # XXX masking in 3d arrays
    pre_occ_densities = numpy.array([numpy.sum(pre_grid[f]) for f in filters]) / float(samp_size[0]*samp_size[1])
    post_nan_densities = numpy.array([numpy.sum(numpy.isnan(post_grid[f])) for f in filters]) / float(samp_size[0]*samp_size[1])
    keep = numpy.where((pre_occ_densities > dens_thresh) & (post_nan_densities == 0))[0]
    print 'keep indices: ', keep
    print 'x and y indices: ', indxs[keep]
    
    return [post_grid[f] for f in filters[keep]], indxs[keep]


    # XXX dropna after gridding
    # XXX interpolate/extrapolate then group/grid? which order?
    # XXX separate into samples 
    # XXX serialize samples or frames
    # XXX store non-spatial parameters separately?


if __name__ == '__main__':
    plac.call(main)
