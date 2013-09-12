import re
import copy
import datetime
import numpy
import glob
import plac
import pandas
import cPickle as pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from solar.util import openz
from solar.learning.models import NeutralPredictor, ARmodel

def plot_weights(n_frames = 1, 
                n_channels = 3,  
                delta_time = 1, 
                size = 9, 
                names = ['tau', 'cloud fraction', 'surface irradiance'],
                center = True,
                l2reg = 1.):

    inputs, targets = load_dataset(
                            size = size,
                            n_frames = n_frames,
                            delta_time = delta_time, # in hours
                            n_channels = n_channels,
                            center = center)
    
    n_dims = size**2

    m = ARmodel(inputs, targets, l2reg = l2reg)
    
    weights = m.theta[:-1] if center else m.theta
    wt_img = numpy.reshape(weights, (n_frames, n_channels, n_dims))

    vmin = numpy.min(wt_img)
    vmax = numpy.max(wt_img)

    grid = ImageGrid(
                    plt.figure(), 111, 
                    nrows_ncols=(n_frames, n_channels), 
                    axes_pad = 0.1,
                    cbar_mode = 'single',
                    cbar_location = 'bottom'
                    )

    for t in xrange(n_frames):
        for i in xrange(n_channels):
            ind = n_frames*t+i
            im = grid[ind].imshow(
                                 numpy.reshape(wt_img[t,i,:], (size,size)), 
                                 vmin = vmin, vmax = vmax,
                                 origin = 'lower',
                                 interpolation = 'nearest'
                                 )
            #grid[ind].title.set_text(names[i], fontsize = 'large')
                
    grid.cbar_axes[0].colorbar(im) #, format='%.2f')
    #grid.xlabel('longitude (0.1 degrees)', fontsize = 'large')
    #grid.ylabel('latitude (0.1 degrees)', fontsize = 'large')
    plt.savefig('data/satellite/output/ar_weights.nf-%i.nc-%i.ws-%i.png' % (n_frames, n_channels, size))

def plot_weights_range(delta_time = 1, sizes = [11], l2reg = 0.5):
    
    for s in sizes:
        plot_weights(delta_time = delta_time, size = s, l2reg = l2reg)

def plot_performance(data_dir = 'data/satellite/output/'):
    
    paths = glob.glob(data_dir + 'ar_performance_tests*.csv')
    paths = sorted(paths)
    
    #print 'using most recent: ', paths[-1]
    #use_path = paths[-1] # most recent
    
    #use_path = data_dir + 'ar_performance_tests.sizes.dt-3.2013-09-05|12:40:05.csv'
    use_path = data_dir + 'ar_performance_tests.sizes.dt-1.2013-09-05|12:05:22.csv'
    df = pandas.read_csv(use_path) 

    grp = re.match('.*ar_performance_tests.*\.dt-(\d+)\..*\.csv',
        use_path).groups()
    delta_time = int(grp[0])
    
    print df 
    # performance v window size
    plt.subplot(211)
    gb = df.groupby(['n-frames','delta-time','n-channels'])
    for indx, group in gb:
        print indx
        print group
        print type(group)
        print 'number of rows: ', len(group.index)
        #print 'ar errors: ', numpy.array(group['ar-test-error'].values[0][1:-1].split(']['), dtype = float) 

        plt.fill_between(group['window-size'].values, 
                        group['ar-test-error'].values - group['ar-test-std'].values, 
                        group['ar-test-error'].values + group['ar-test-std'].values, 
                        alpha=0.1, linewidth=0, color = 'g')
        plt.plot(group['window-size'], group['ar-test-error'], label = ' AR test', color = 'g')

        plt.fill_between(group['window-size'].values, 
                        group['ar-train-error'].values - group['ar-train-std'].values, 
                        group['ar-train-error'].values + group['ar-train-std'].values, 
                        alpha=0.1, linewidth=0, color = 'b')
        plt.plot(group['window-size'], group['ar-train-error'], label = ' AR train', color = 'b')

        plt.fill_between(group['window-size'].values, 
                        group['np-test-error'].values - group['np-test-std'].values, 
                        group['np-test-error'].values + group['np-test-std'].values, 
                        alpha=0.1, linewidth=0, color = 'r')
        plt.plot(group['window-size'], group['np-test-error'], label = ' Eulerian test', color = 'r')
        
    plt.legend(loc = 'center right')
    plt.suptitle('%i Hour Forecast Performance v. Window Size' % delta_time, fontsize = 'large')
    plt.xlabel('window size', fontsize = 'large')
    plt.ylabel('RMSE $W/m^2$', fontsize = 'large')

    plt.subplot(212)

    for indx, group in gb:
        print indx
        print group
        print type(group)
        print 'number of rows: ', len(group.index)
        #print 'ar errors: ', numpy.array(group['ar-test-error'].values[0][1:-1].split(']['), dtype = float) 

        #plt.fill_between(group['window-size'].values, 
        #                group['ar-test-error'].values - group['ar-test-std'].values, 
        #                group['ar-test-error'].values + group['ar-test-std'].values, 
        #                alpha=0.1, linewidth=0, color = 'g')
        plt.plot(group['window-size'], 100*(group['np-test-error'].values - group['ar-test-error'].values) / group['np-test-error'].values  , label = ' AR test', color = 'g')

        #plt.fill_between(group['window-size'].values, 
        #                group['ar-train-error'].values - group['ar-train-std'].values, 
        #                group['ar-train-error'].values + group['ar-train-std'].values, 
        #                alpha=0.1, linewidth=0, color = 'b')
        plt.plot(group['window-size'], 100*(group['np-train-error'].values - group['ar-train-error'].values) / group['np-test-error'].values, label = ' AR train', color = 'b')

        #plt.fill_between(group['window-size'].values, 
        #                group['np-test-error'].values - group['np-test-std'].values, 
        #                group['np-test-error'].values + group['np-test-std'].values, 
        #                alpha=0.1, linewidth=0, color = 'r')
        #plt.plot(group['window-size'], group['np-test-error'], label = ' NP test', color = 'r')
    
    #plt.title('Autoregression Performance v. Window Size')
    plt.xlabel('window size', fontsize = 'large')
    plt.ylabel('% better than Eulerian', fontsize = 'large')
    plt.legend(loc = 'lower right', fontsize = 'large')

    
    # performance v number of frames
    #plt.subplot(311,2)
    #gb = df.groupby(['window-size','delta-time','n-channels'])
    #for indx, group in gb:
    #    plt.plot(df['n-frames'], df['ar-test-error'], label = str(indx)+' AR test') 
    #    plt.plot(df['n-frames'], df['ar-train-error'], label = str(indx)+' AR train') 
    #    plt.plot(df['n-frames'], df['np-test-error'], label = str(indx)+' NP test') 
    #    plt.plot(df['n-frames'], df['np-train-error'], label = str(indx)+' NP train') 
    #plt.legend()
    #
    ## performance v forecast interval (delta time)
    #plt.subplot(311,3)
    #gb = df.groupby(['window-size','n-frames','n-channels'])
    #for indx, group in gb:
    #    plt.plot(df['delta-time'], df['ar-test-error'], label = str(indx)+' AR test') 
    #    plt.plot(df['delta-time'], df['ar-train-error'], label = str(indx)+' AR train')
    #    plt.plot(df['delta-time'], df['np-test-error'], label = str(indx)+' NP test')
    #    plt.plot(df['delta-time'], df['np-train-error'], label = str(indx)+' NP train')
    #plt.legend()

    plt.savefig(use_path[:-4] + '.png')


def load_dataset(
    data_dir = 'data/satellite/processed/', 
    target_channel = 'swd_sfc',
    size = 9,
    n_frames = 1,
    delta_time = 1., # in hours
    n_channels = 3,
    center = True):

    ''' loads dataset from data_dir using all files that match the given 
    parameters returns accumulated input and target arrays '''
    
    # XXX gzip?
    paths = glob.glob(data_dir + 'goes-insolation.dt-%0.1f.nf-%i.nc-%i.ws-%i.str-*.dens-*.nsamp-*.0.pickle' % 
            (delta_time, n_frames, n_channels, size))

    assert len(paths) > 0
    sorted(paths)
    print 'matching files: ', paths

    print 'using: ', paths[0]
    
    # find all files from the same batch
    path = paths[0]
    part = path[:-9] # XXX 11 if zipped
    paths = glob.glob(part+'*.pickle')
    
    print 'len of paths: ', len(paths)

    par_grp = re.match(
        '.*goes.*\.nf-(\d+)\.nc-(\d+)\.ws-(\d+).*\.nsamp-(\d+)\.\d+\.pickle',
        path).groups()
    n_frames, n_channels, wind_size, n_samples = map(int, par_grp)
    n_wind_cells = wind_size**2    

    dataset = Dataset(n_samples, n_frames, n_channels, n_wind_cells)
        
    print 'min cpus for pool: ', min(mp.cpu_count, len(paths))
    
    pool = mp.Pool(min(mp.cpu_count(), len(paths)))
    print 'paths: ', paths
    for p in paths:
        print 'opening file: ', p 
        pool.apply_async(unpack_dataset, args = [p], callback=dataset.accumulate)
        #dataset.accumulate(apply(unpack_dataset, [p]))
    
    pool.close()
    pool.join()

    # did we fill all the spaces we expected?
    assert not (numpy.isnan(dataset.inputs)).any()
    assert not (numpy.isnan(dataset.targets)).any()
    #assert not (numpy.isnan(dataset.timestamps)).any()
    assert n_samples == dataset.data['n-samples']
    
    print 'files loaded'

    input_names = dataset.data['input-names']
    target_index = input_names.index(target_channel)
    targets = dataset.targets[:,target_index]
    mn,std = dataset.data['norm-stats'][target_index,:]
    inputs = dataset.inputs
    
    n_samples, nf, nc, n_dims = inputs.shape
    assert nf == n_frames
    assert nc == n_channels
    assert size**2 == n_dims

    return inputs, targets



