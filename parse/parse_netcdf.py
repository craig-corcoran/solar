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

def cdf_to_frame(path):
    
    ds = cdf.Dataset(path)
    
    # xxx
    max_rows = max(map(lambda x: len(get_data_from_var(x)[0]), 
                       ds.variables.values()))

    df = pandas.DataFrame(index = numpy.arange(max_rows))
    meta = pandas.DataFrame()
    
    for head,var in ds.variables.items():
        
        data, mask = get_data_from_var(var)
        
        if len(data) == 1:
            data = data.repeat(max_rows)
        else: 
            assert len(data) == max_rows

        if len(df.index)==0:
            assert False
            df = pandas.DataFrame(data, columns = [head])
        else: 
            df[head] = data
        
        # set masked values to NaN instead?
        if mask is not None: df[head + '_mask'] = mask
        
        # collect meta data for variable
        row = pandas.DataFrame([var.__dict__])
        meta = pandas.concat([meta, row], ignore_index=True)

    meta.index = pandas.Index(ds.variables.keys())

    return df, meta

@plac.annotations(path = 'path to netCDF (.nc) file')
def main(path = 'data/satellite/download.class.ngdc.noaa.gov/download/123411444/001/', #'satellite/sample/gsipL3_g13_GEDISK_2013101_2045.nc',
         frac = 0.75):
    ''' script for parsing GOES netCDF (.nc) data into pandas data frame, then
    plotting num (before cleaning) samples for all spatial data '''
    
    paths = glob.glob(path + '*.nc')
    paths = sorted(paths, reverse = True)

    for n, p in enumerate(paths):
        
        df, meta = cdf_to_frame(p)
        x = df['lon_cell']
        y = df['lat_cell']
        
        #x[x<-180] = x[x<-180] + 180. # ever needed for y?
        #x[x>180] = x[x>180] - 180. # ever needed for y?
        x[x < 0] = x[x < 0] + 360.
        
        extrema = {}
        extrema['min'] = numpy.array([numpy.min(x), numpy.min(y)]).astype(float)
        extrema['max'] = numpy.array([numpy.max(x), numpy.max(y)]).astype(float)
        delta = (extrema['max'] - extrema['min']) / 2.

        print 'delta: ', delta
        print 'min: ', extrema['min']
        print 'max: ', extrema['max']


        center = extrema['min'] + delta
        center0 = numpy.array((numpy.mean(x), numpy.mean(y))) # use min max instead?    
        print 'center: ', center
        print 'alt center: ', center0

        bounds = {}
        bounds['min'] = center - frac * delta
        bounds['max'] = center + frac * delta

        inc = numpy.where((x > bounds['min'][0]) & (x < bounds['max'][0]) & 
                            (y > bounds['min'][1]) & (y < bounds['max'][1]))[0] 

        #x_inds = set(numpy.where((x > bounds['min'][0]) & (x < bounds['max'][0]))[0])
        #y_inds = set(numpy.where((y > bounds['min'][1]) & (y < bounds['max'][1]))[0])
        #include = list(x_inds.intersection(y_inds))

        #print 'includes the same: ', (inc == include).all()

        x_inc = x[inc]
        y_inc = y[inc]
        
        pyplot.clf()
        pyplot.plot(x,y,'o')
        pyplot.autoscale(tight=True)
        pyplot.savefig('plots/lat-lon%i.pdf' % n)

        pyplot.clf()
        pyplot.plot(x_inc,y_inc,'o')
        pyplot.savefig('plots/lat-lon-included%i.pdf' % n)



    

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
    arr = var[:]
    if type(arr) is numpy.ma.core.MaskedArray:
        return arr.data.T, arr.mask.T
    elif type(arr) is numpy.ndarray:
        return arr.T, None

if __name__ == '__main__':
    plac.call(main)
