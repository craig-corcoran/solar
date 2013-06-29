import plac
import glob
import numpy
import pandas
import netCDF4 as cdf
import matplotlib.pyplot as pyplot
#from mpl_toolkits.mplot3d import Axes3D

#gsipL3_g13_GEDISK_2013101_2045.nc
#gsipL3_g13_GENHEM_2013101_1945.nc


# XXX whats the difference between missing and mask?

def cdf_to_frame(path, inputs, dropna_thresh = 2):
    
    ds = cdf.Dataset(path)
    
    max_rows = max(ds.variables[ds.variables.keys()[-1]].shape)

    df = pandas.DataFrame(index = numpy.arange(max_rows))
    meta = pandas.DataFrame()
    
    for head,var in ds.variables.items():
        
        data, mask = get_data_from_var(var)
        
        if len(data) == 1:
            # store non-spatial parameters?
            data = data.repeat(max_rows)
        else: 
            assert len(data) == max_rows 
        
        if 'SCALED_MISSING' in var.ncattrs():
            missing_val = var.SCALED_MISSING
            data[data == missing_val] = numpy.nan

        if mask is not None: 
            data[mask] = numpy.nan
        #if mask is not None: df[head + '_mask'] = mask

        df[head] = data
        
        # collect meta data for variable
        row = pandas.DataFrame([var.__dict__])
        meta = pandas.concat([meta, row], ignore_index=True)
    

    meta.index = pandas.Index(ds.variables.keys())

    # drop rows with more than two missing inputs
    print 'len before dropping na: ', len(df)
    df = df.dropna(thresh=(len(inputs)-dropna_thresh), subset=inputs)
    print 'len after dropping na: ', len(df)
    df.index = pandas.Index(xrange(len(df))) # reindex 

    return df, meta

# XXX spatial density - histogram
# grid binning
@plac.annotations(path = 'path to netCDF (.nc) file')
def main(
    path = 'data/satellite/download.class.ngdc.noaa.gov/download/123411444/001/',
    dlat = 5.,
    dlon = 5.,
    sample_size = (5,5,2), # (dlat, dlon, dtime)
    inputs = ['ch2','ch2_cld','ch2_std',
            'ch9',
            'ch14','ch14_std','ch14clr', 
            'frac_snow',
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

    dlatlon = numpy.array([dlat, dlon])
    paths = glob.glob(path + '*.nc')
    #paths = sorted(paths, reverse = True)
    

    # XXX group by (year, day, time, lat, lon)
    # XXX switch to lat, lon
    # XXX need grid? just use pandas df?

    inf = numpy.array([numpy.inf, numpy.inf])
    extrema = {'min': inf, 'max': -inf}
    clean_frames = [None]*len(paths)
    for i,p in enumerate(paths[-2:]):
        
        df, meta = cdf_to_frame(p, inputs)
        x = df['lon_cell']
        y = df['lat_cell']
        x[x < 0] = x[x < 0] + 360.
        df['lon_cell'] = x
        clean_frames[i] = df
        
        samp_min = numpy.array([numpy.min(x), numpy.min(y)]).astype(float)
        samp_max = numpy.array([numpy.max(x), numpy.max(y)]).astype(float)
        extrema['min'] = numpy.min(numpy.vstack([extrema['min'],samp_min]), axis=0) 
        extrema['max'] = numpy.max(numpy.vstack([extrema['max'],samp_max]), axis=0)
        
    delta = (extrema['max'] - extrema['min'])
    grid_dim = (delta / dlatlon).astype(int) + 1
    grid   = numpy.array([[None]*grid_dim[1]]*grid_dim[0])
    #counts = numpy.array([[None]*grid_dim[1]]*grid_dim[0])
    #nones  = numpy.array([[None]*grid_dim[1]]*grid_dim[0])
    
    print 'bounding rect size: ', delta
    print 'min: ', extrema['min']
    print 'max: ', extrema['max']

    all_data = pandas.concat(clean_frames, ignore_index = True)
    x = all_data['lon_cell']
    y = all_data['lat_cell']
    indx = numpy.vstack([x - extrema['min'][0],y - extrema['min'][1]])
    indx = numpy.floor(indx / dlatlon[:,None]).astype(int)
    all_data['lon_index'] = indx[0]
    all_data['lat_index'] = indx[1]

    grouped = all_data.groupby(['lon_index','lat_index'])
    mean_data = grouped.mean().reset_index()
    
    print mean_data['lon_index'].values.shape
    print mean_data[inputs].values.shape
    grid[mean_data['lon_index'], mean_data['lat_index']] = mean_data[inputs].values

    print grid


    
    # use grouby and aggregate instead
    #for n, p in enumerate(paths[-2:]):

        #print 'path: ', p
        
        #df = clean_frames[n]
        #dat_vecs = df[inputs]

        #x = df['lon_cell']
        #y = df['lat_cell']

        #indx = numpy.vstack([x - extrema['min'][0],y - extrema['min'][1]])
        #indx = numpy.floor(indx / dlatlon[:,None]).astype(int)

        #for i in xrange(len(df.index)):
        
            ## XXX don't add nan
            #row = dat_vecs.ix[i]
            #is_nan = row.isnull()
            #row[is_nan] = 0.
            #row = row.values
            #count = (-is_nan).astype(int).values

            #if grid[indx[0][i], indx[1][i]] is None:
                #grid[indx[0][i], indx[1][i]] = row
                #counts[indx[0][i], indx[1][i]] = count
            #else:
                #assert counts[indx[0][i], indx[1][i]] is not None
                #grid[indx[0][i], indx[1][i]] += row
                #counts[indx[0][i], indx[1][i]] += count

            ##if (i % 10000) == 0:
                ##print 'index: ', i
                ##print 'row added: ', grid[indx[0][i], indx[1][i]]

    #print counts.shape
    #print nones.shape
    #print counts
    #print nones
    #mask = -(counts == nones)
    #print mask
    #print grid[mask]
    #print grid.shape
    
    #grid[mask] = grid[mask] / counts[mask].astype(float)
            
    #print grid
    

@plac.annotations(path = 'path to netCDF (.nc) file')
def plots(path = 'data/satellite/download.class.ngdc.noaa.gov/download/123411444/001/', #'satellite/sample/gsipL3_g13_GEDISK_2013101_2045.nc',
         num = 25000):
    ''' script for parsing GOES netCDF (.nc) data into pandas data frame, then
    plotting num (before cleaning) samples for all spatial data '''
    
    paths = glob.glob(path + '*.nc')
    #paths = sorted(paths)

    for n, p in enumerate(paths):
        print p
        
        ds = cdf.Dataset(p)
        max_rows = max(map(lambda x: len(get_data_from_var(x)[0]), 
                           ds.variables.values()))
        df = pandas.DataFrame(index = numpy.arange(max_rows))
        meta = pandas.DataFrame()
        
        to_plot = [] # list of spatial vars for plotting
        for head,var in ds.variables.items():
            
            data, mask = get_data_from_var(var)
            
            if len(data) == 1:
                data = data.repeat(max_rows)
            else: 
                assert len(data) == max_rows
                to_plot.append(head)

            if len(df.index)==0:
                df = pandas.DataFrame(data, columns = [head])
            else: 
                df[head] = data
            
            # set masked values to NaN instead?
            if mask is not None: df[head + '_mask'] = mask
            
            # collect meta data for variable
            row = pandas.DataFrame([var.__dict__])
            meta = pandas.concat([meta, row], ignore_index=True)

        meta.index = pandas.Index(ds.variables.keys())
        print 'meta data: ', meta

        #perm = numpy.random.permutation(max_rows)
        #perm = numpy.arange(max_rows) # unmixed
        for i,col in enumerate(to_plot):
            print 'plotting %s: %s, %i of %i' % \
                    (col, meta.loc[col]['long_name'], i+1, len(to_plot))
            #x = df['lon_cell'][perm]#[:num]
            #y = df['lat_cell'][perm]#[:num]
            #z = df[col][perm]#[:num]
            
            pyplot.clf()
            x = df['lon_cell']
            y = df['lat_cell']
            pyplot.plot(x,y,'.')
            pyplot.autoscale(tight=True)
            pyplot.savefig('plots/lat-lon%i.pdf' % i)

            
            #num_lat = numpy.where(y != y[0])[0][0] + 1
            #print 'num lat: ', num_lat
            #print 'array length: ', len(x)
            #print 'div: ', len(x) / float(num_lat)
            


            #print numpy.array(x), ' ,', numpy.array(y)
        
            # remove missing variables
            #missing = meta.loc[col]['SCALED_MISSING']
            #not_missing = (z != missing) # mask array
            
            #missing = (z == missing) # mask array
            #try:
                #mask = numpy.array(df[col + '_mask'][perm][:num])
                #print len(missing)
                #print len(mask)
                #print 'missing and mask: ', (missing == mask).all()
            #except KeyError as e:
                #pass

            #x,y,z = map(lambda lam: lam[not_missing], [x,y,z])

            #print '%i of %i are not missing' % (len(z), num)
            
            # plot and save
            #pyplot.scatter(x,y,c=z, cmap=pyplot.cm.jet)
            #pyplot.savefig('plots/%s-%i-%i.pdf' % (col, n, num))
            #pyplot.clf()
    
def plot_3d(x, y, z, file_name = None):
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=pyplot.cm.jet, linewidth=0.2)
    
    if file_name is None:
        pyplot.show()
    else:
        pyplot.savefig(file_name)

def get_data_from_var(var):
    ''' unpacks a netCDF Variable and returns the array of data and the mask if
    the variable is a masked array, otherwise None'''
    arr = var[:].astype(float)
    if type(arr) is numpy.ma.core.MaskedArray:
        return arr.data.T, arr.mask.T
    elif type(arr) is numpy.ndarray:
        return arr.T, None

if __name__ == '__main__':
    plac.call(main)
