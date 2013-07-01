import plac
import glob
import numpy
import pandas
import itertools as it
import netCDF4 as cdf
from scipy import interpolate
import matplotlib.pyplot as pyplot

# XXX whats the difference between missing and mask?

def get_data_from_var(var):
    ''' unpacks a netCDF Variable and returns the array of data and the mask if
    the variable is a masked array, otherwise None'''
    arr = var[:].astype(float)
    if type(arr) is numpy.ma.core.MaskedArray:
        return arr.data.T, arr.mask.T
    elif type(arr) is numpy.ndarray:
        return arr.T, None

def cdf_to_frame(path, inputs, dropna_thresh = None):
    
    ds = cdf.Dataset(path)
    
    max_rows = max(ds.variables[ds.variables.keys()[-1]].shape)

    df = pandas.DataFrame(index = numpy.arange(max_rows))
    meta = pandas.DataFrame()
    
    for head,var in ds.variables.items():
        
        data, mask = get_data_from_var(var)
        
        if len(data) == 1:
            data = data.repeat(max_rows)
        else: 
            assert len(data) == max_rows 
        
        # set missing values to nan
        if 'SCALED_MISSING' in var.ncattrs():
            missing_val = var.SCALED_MISSING
            data[data == missing_val] = numpy.nan
        
        # set masked values to nan
        if mask is not None: 
            data[mask] = numpy.nan
        #if mask is not None: df[head + '_mask'] = mask

        df[head] = data
        
        # collect meta data for variable
        row = pandas.DataFrame([var.__dict__])
        meta = pandas.concat([meta, row], ignore_index=True)
    

    meta.index = pandas.Index(ds.variables.keys())

    # drop rows with more than two missing inputs
    if dropna_thresh is not None:
        print 'len before dropping na: ', len(df)
        df = df.dropna(thresh=(len(inputs)-dropna_thresh), subset=inputs)
        print 'len after dropping na: ', len(df)

    return df, meta

def convert_to_datetime(img_date, img_time):
    
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

# swd, frac_ice/water/total, tau, olr, 
@plac.annotations(path = 'path to netCDF (.nc) file')
def main(
    path = 'data/satellite/download.class.ngdc.noaa.gov/download/123411444/001/',
    dlat = .5,
    dlon = .5,
    sample_size = (5,5,2), # (dlat, dlon, dtime)
    interp_method = 'cubic',
    inputs = ['ch2','ch2_cld','ch2_std',
            'ch9',
            'ch14','ch14_std','ch14clr', 
            #'frac_snow',
            'lst', 'trad',
            'iwp', 'lwp',
            'cld_temp','cld_press',
            'tau',
            'frac_total_cld', 'frac_ice_cld', 'frac_water_cld',
            'ch2_ccr', 'ch2_ccr_std',
            'olr', 
            'swd_sfc','swu_sfc',
            'swd_toa','swu_toa',
            'swd_sfc_clr', 'swd_sfc_clr', 'swu_toa_clr',
            'vis_down_sfc', 
            #'cld_type',
            'ozone',
            'tpw',
            'lwdsfc',
            'lwusfc']):

    paths = glob.glob(path + '*.nc')
    paths = sorted(paths, reverse = True)

    dlatlon = numpy.array([dlat, dlon])
    
    # read netcdf file
    clean_frames = [None]*len(paths[:4])
    for i,p in enumerate(paths[:4]):
        # read into pandas data frame
        df, meta = cdf_to_frame(p, inputs)

        # prevent longitude wrapping
        x = df['lat_cell']
        y = df['lon_cell']
        y[y < 0] = y[y < 0] + 360.
        df['lon_cell'] = y
        clean_frames[i] = df 

    all_data = pandas.concat(clean_frames, ignore_index = True)
    pos_extrema = {'max': numpy.max([all_data['lat_cell'].values, 
                        all_data['lon_cell'].values], axis = 1),
                    'min': numpy.min([all_data['lat_cell'].values, 
                        all_data['lon_cell'].values], axis = 1)}
    
    # index lat and lon according to range and dlat, dlon
    x = all_data['lat_cell']
    y = all_data['lon_cell']
    indx = numpy.vstack([x - pos_extrema['min'][0],y - pos_extrema['min'][1]])
    indx = numpy.floor(indx / dlatlon[:,None]).astype(int)
    all_data['lat_index'] = indx[0]
    all_data['lon_index'] = indx[1]
    
    #print 'plotting density histogram'
    #pyplot.clf()
    #pyplot.hexbin(x, y)
    #pyplot.savefig('plots/satellite/density.pdf')
    
    # average over lat,lon and time groups
    grouped = all_data.groupby(['lat_index','lon_index', 'img_date', 'img_time'])
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

    # XXX dropna after gridding
    # XXX interpolate/extrapolate then group/grid? which order?
    # XXX separate into samples 
    # XXX serialize samples or frames
    # XXX store non-spatial parameters separately?


if __name__ == '__main__':
    plac.call(main)
