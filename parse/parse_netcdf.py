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
    
    def __init__(self, path_regex, inputs, lat_range = None, lon_range = None, 
                interp_buffer = None, use_masked = True):
        
        self.inputs = inputs
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.interp_buffer = interp_buffer

        paths = glob.glob(path_regex)
        assert len(paths) > 0
        
        # read netcdf file
        clean_frames = [None]*len(paths)
        for i,p in enumerate(paths):
            # read into pandas data frame
            df, meta = self.nc_to_frame(p, use_masked = use_masked) # removes all but lat, lon, inputs, and datetime
            clean_frames[i] = df 

        self.meta = meta # should be the same across files

        print 'meta data: ', meta
        print 'concatenating all files into one frame'

        # combine files into one frame
        self.frame = pandas.concat(clean_frames, ignore_index = True) 

        # add datetime column
        self.frame['datetime'] = self.convert_to_datetime(
                                            self.frame['img_date'].values,
                                            self.frame['img_time'].values)
        #self.frame = self.frame.sort('datetime')
        
        print 'centering and rescaling data'
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
        self.mean_vec = self.frame[self.inputs].mean()
        self.std_vec = self.frame[self.inputs].std()

        print 'mean: ', numpy.array(self.mean_vec)
        print 'std: ', numpy.array(self.std_vec)

        #self.frame[self.inputs] = self.frame[self.inputs].astype(float) - self.mean_vec # XXX
        #self.frame[self.inputs] = self.frame[self.inputs].astype(float) / self.std_vec

    # XXX unscale
    # convert theta/phi to lat/lon

    @staticmethod
    def latlon_to_thetaphi(lat, lon):
        # convert lat to 0 to pi
        lat = (lat+90.)*(numpy.pi/180.)

        # convert lon to 0 to 2pi
        lon = copy.deepcopy(lon)
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
        
        #print 'NetCDF keys: ', ds.variables.keys()
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
    path = 'data/satellite/download.class.ngdc.noaa.gov/download/123483484/001/gsipL3_g13_GENHEM_2013121_01', #  'data/satellite/gsipL3_g13_GENHEM_2013121_0*', #
    sample_shape = (50,50,2), # (n_lat, n_lon, n_time)
    lat_range = None, #(41, 44), # (10., 13.), #(34., 37.), #(-50, 50), # , #
    lon_range = None, #(-110, -105), # (-100., -95.), #(170, 280), # (95, 100), #oklahoma
    interp_buffer = (2,2),
    use_masked = True,
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
    
    gd = GoesData(path+'*.nc', inputs, lat_range, lon_range, interp_buffer, 
            use_masked = use_masked)

    print gd.frame
    print 'new lat/lon bounds: '
    pos_extrema = {'max': numpy.max([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1),
                   'min': numpy.min([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1)}
    
    print 'lat/lon extrema: ', pos_extrema 
    print 'datetimes: ', set(gd.frame['datetime'])
    


    for i, dt in enumerate(set(gd.frame['datetime'])):
        
        time_slice = gd.frame[gd.frame['datetime'] == dt]
        
        for inp in inputs:
            no_missing = time_slice.dropna(how='any', subset=[inp])
            print 'length of clean points for %s: ' % inp, len(no_missing)
            if len(no_missing) > 0:
                lat = no_missing['lat_rad'].values
                lon = no_missing['lon_rad'].values
                val = no_missing[inp].values
                
                print 'clean lat max min: ', numpy.max(lat), numpy.min(lat)
                print 'clean lon max min: ', numpy.max(lon), numpy.min(lon)
              
                #spline_built = False
                #mult = 0.5
                #while not spline_built:
                    #try:
                        #print 'building spline representation' 
                        #spline = interpolate.SmoothSphereBivariateSpline(lat, lon, val, 
                                                                #s=mult*len(no_missing))
                        #spline_built = True
                    #except ValueError as e:
                        #print e
                        #mult += 0.2
                        #print 'increasing s value to %f times number of points' % mult
                
                frac = 0.1 # remove this fraction of the image from the borders for interpolation
                dlat = numpy.max(lat) - numpy.min(lat)
                dlon = numpy.max(lon) - numpy.min(lon)
                #ilat = numpy.linspace(numpy.min(lat) + frac/2.*dlat, numpy.max(lat) - frac/2.*dlat, sample_shape[0])
                #ilon = numpy.linspace(numpy.min(lon) + frac/2.*dlon, numpy.max(lon) - frac/2.*dlon, sample_shape[1])
                #ilat_grid, ilon_grid = it.izip(*it.product(ilat, ilon))
                lat_grid, lon_grid = numpy.mgrid[
                    numpy.min(lat)+frac/2.*dlat : numpy.max(lat)-frac/2.*dlat : sample_shape[0]*1j,
                    numpy.min(lon)+frac/2.*dlon : numpy.max(lon)-frac/2.*dlon : sample_shape[1]*1j]
                
                print 'interpolating to grid'
                interpol = interpolate.griddata((lat,lon), val, (lat_grid, lon_grid), method = 'cubic')
                #interpol = spline(ilat, ilon)
                #irbf = interpolate.Rbf(lat, lon, val, smooth = 1e-8)
                #interpol = irbf(lat_grid, lon_grid)

                print 'interpolated data shape: ', interpol.shape
                print 'max, min clean interpolated vals: ', numpy.max(interpol), numpy.min(interpol)

                print 'plotting'
                pyplot.clf()
                ax = pyplot.subplot(121)
                n_plot = 10000
                if len(lat) > n_plot:
                    print 'subsampling'
                    perm = numpy.random.permutation(len(lat))
                    latplot = lat[perm][:n_plot]
                    lonplot = lon[perm][:n_plot]
                    valplot = val[perm][:n_plot]
                else:
                    latplot = lat
                    lonplot = lon
                    valplot = val

                vmin = numpy.min([numpy.min(valplot), numpy.min(interpol)])
                vmax = numpy.max([numpy.max(valplot), numpy.max(interpol)])
                
                if numpy.isnan(vmin) or numpy.isnan(vmax):
                    pyplot.scatter(latplot, lonplot, c = valplot, 
                        cmap = 'jet', s=10, linewidths = 0)
                else:
                    pyplot.scatter(latplot, lonplot, c = valplot, 
                        cmap = 'jet', s=10, linewidths = 0, vmin=vmin, vmax=vmax)
                    pyplot.colorbar()

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                ax = pyplot.subplot(122)
                if numpy.isnan(vmin) or numpy.isnan(vmax): 
                    pyplot.scatter(lat_grid, lon_grid, c = interpol, 
                        cmap = 'jet', s=10, linewidths = 0)
                else:
                    pyplot.scatter(lat_grid, lon_grid, c = interpol, 
                        cmap = 'jet', s=10, linewidths = 0, vmin=vmin, vmax=vmax)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                pyplot.savefig(('interp-%s-%s-%i-%s.pdf' % 
                        (inp, 'use_masked' if use_masked else '', i, str(dt))).replace(' ', ''))

def parse_imager(
    path ='data/satellite/download.class.ngdc.noaa.gov/download/123483494/001/goes13.2013.121',
    sample_shape = (50,50,2), # (n_lat, n_lon, n_time)
    lat_range = None, #(34., 37.), #(-50, 50), # , #
    lon_range = None, #(-100., -95.), #(170, 280), # (95, 100), #oklahoma
    interp_buffer = (2,2),
    use_masked = True,
    inputs = [
            u'version', 
            u'sensorID', 
            u'imageDate', 
            u'imageTime', 
            u'startLine', 
            u'startElem', 
            u'time', 
            u'dataWidth', 
            u'lineRes', 
            u'elemRes', 
            u'prefixSize', 
            u'crDate', 
            u'crTime', 
            u'bands', 
            #u'auditTrail', 
            u'data', 
            u'lat', 
            u'lon']):
    
    gd = GoesData(path+'*.nc', inputs, lat_range, lon_range, interp_buffer, 
            use_masked = use_masked)

    print gd.frame
    print 'new lat/lon bounds: '
    pos_extrema = {'max': numpy.max([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1),
                   'min': numpy.min([gd.frame['lat_cell'].values, 
                        gd.frame['lon_cell'].values], axis = 1)}
    
    print 'lat/lon extrema: ', pos_extrema 
    print 'datetimes: ', gd.frame['datetime']


if __name__ == '__main__':
    plac.call(main)
