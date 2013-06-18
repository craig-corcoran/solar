import plac
import numpy
import pandas
import netCDF4 as cdf
import matplotlib.pyplot as pyplot
#from mpl_toolkits.mplot3d import Axes3D

#gsipL3_g13_GEDISK_2013101_2045.nc
#gsipL3_g13_GENHEM_2013101_1945.nc
@plac.annotations(path = 'path to netCDF (.nc) file')
def main(path = 'satellite/sample/gsipL3_g13_GEDISK_2013101_2045.nc',
         num = 25000):
    ''' script for parsing GOES netCDF (.nc) data into pandas data frame, then
    plotting num (before cleaning) samples for all spatial data '''
    ds = cdf.Dataset(path)
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

    perm = numpy.random.permutation(max_rows)
    #perm = numpy.arange(max_rows) # unmixed
    for i,col in enumerate(to_plot):
        print 'plotting %s: %s, %i of %i' % \
                (col, meta.loc[col]['long_name'], i+1, len(to_plot))
        x = df['lon_cell'][perm][:num]
        y = df['lat_cell'][perm][:num]
        z = df[col][perm][:num]
    
        # remove missing variables
        missing= meta.loc[col]['SCALED_MISSING']
        not_missing = (z != missing) # mask array
        x,y,z = map(lambda lam: lam[not_missing], [x,y,z])

        print '%i of %i are not missing' % (len(z), num)
        
        # plot and save
        pyplot.scatter(x,y,c=z, cmap=pyplot.cm.jet)
        pyplot.savefig('plots/%s-%i.pdf' % (col, num))
        pyplot.clf()
    
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
