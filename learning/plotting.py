import re
import numpy
import glob
import plac
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from solar.learning.models import ARmodel
from solar.learning.goes_experiment import load_dataset

@plac.annotations(
n_frames = ('number of time lags into the past used for them model whose weights are plotted', 'option', None, int),
delta_time = ('forecast horizon in hours', 'option', None, float),
size = ('window size, assumed to be square size x size', 'option', None, int),
names = ('list of strings to use for input channels', 'option', None, None),
center = ('boolean value, should a constant feature be added, centering the data (not plotted)', 'option', None, bool),
l2reg = ('l2 regularization constant', 'option', None, float)
)
def plot_weights(n_frames = 1, 
                n_channels = 3,  
                delta_time = 1, 
                size = 9, 
                names = ['tau', 'cloud fraction', 'surface irradiance'],
                center = True,
                l2reg = 1.):

    ''' learn a spatiotemporal model using the given parameters, then plot the
    resulting weights'''

    n_channels = len(names)

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

@plac.annotations(
data_dir = ('directory in which the csv performance file is stored', 'option', None, str),
plot_file = ('particular csv file to use, if None last file in alphanumeric order is used (will be most recent, except prefer longer intervals)', 'option', None, float)
)
def plot_performance(data_dir = 'data/satellite/output/', plot_file = None):
    ''' read a window size v performance csv file and plot the corresponding data'''
    paths = glob.glob(data_dir + 'ar_performance_tests*.csv')
    paths = sorted(paths)
    
    if plot_file is None:
        print 'using last file: ', paths[-1]
        use_path = paths[-1] # most recent
    
    else:
        print 'using specified file: ', plot_file
        use_path = data_dir + plot_file
        
    df = pandas.read_csv(use_path) 

    grp = re.match('.*ar_performance_tests.*\.dt-(\d+)\..*\.csv',
        use_path).groups()
    delta_time = int(grp[0])
    
    # performance v window size
    plt.subplot(211)
    gb = df.groupby(['n-frames','delta-time','n-channels'])
    for indx, group in gb:
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
        plt.plot(group['window-size'], 100*(group['np-test-error'].values - group['ar-test-error'].values) / group['np-test-error'].values  , label = ' AR test', color = 'g')

        plt.plot(group['window-size'], 100*(group['np-train-error'].values - group['ar-train-error'].values) / group['np-test-error'].values, label = ' AR train', color = 'b')
    
    plt.xlabel('window size', fontsize = 'large')
    plt.ylabel('% better than Eulerian', fontsize = 'large')
    plt.legend(loc = 'lower right', fontsize = 'large')
    plt.savefig(use_path[:-4] + '.png')

if __name__ == '__main__':
    
    #plac.call(plot_performance)
    plac.call(plot_weights)

