import datetime
import numpy
import glob
import plac
import pandas
import cPickle as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from solar.util import openz


class NeutralPredictor(object):

    def __init__(self, input_array, target_array, center = None):
        
        self.n_dims = input_array.shape[-1]
        self.theta = numpy.zeros(input_array.shape[1:])
        self.theta[-1,:,numpy.ceil(self.n_dims/2.)] = 1.
        
    def make_prediction(self, input_array):
        
        assert input_array.ndim == 4
        return input_array[:,-1,:,numpy.ceil(self.n_dims/2.)] # n_samples, n_frames, n_channels, n_dims

    def test_prediction(self, input_array, target_array):
        
        return numpy.sqrt(numpy.mean(
                  (self.make_prediction(input_array)-target_array)**2,
                  axis=0)) 

class ARmodel(NeutralPredictor):
    
    def __init__(self, input_array, target_array, l2reg = 0., center = True):
        
        self.n_samples = input_array.shape[0]

        if len(input_array.shape) > 2:
            input_array = self._flatten_samples(input_array)
        
        self.center = center
        if self.center:
            input_array = numpy.hstack([input_array, numpy.ones((input_array.shape[0],1))])
        
        #self.theta = numpy.linalg.lstsq(input_array, target_array)[0]
        self.l2reg = l2reg
        reg_matrix = self.l2reg * numpy.eye(input_array.shape[1])
        if center:
            reg_matrix[-1,-1] = 0.
        C = (1./self.n_samples) * numpy.dot(input_array.T, input_array) + reg_matrix
        b = (1./self.n_samples) * numpy.dot(input_array.T, target_array)
        self.theta = numpy.linalg.solve(C, b)
    
    @staticmethod
    def _flatten_samples(input_array):
    
        new_shape = (input_array.shape[0], numpy.prod(input_array.shape[1:]))
        return numpy.reshape(input_array, new_shape)

    def make_prediction(self, input_array):
        
        if len(input_array.shape) > 2:
            input_array = self._flatten_samples(input_array)

        if self.center:
            input_array = numpy.hstack([input_array, numpy.ones((input_array.shape[0],1))])

        return numpy.dot(input_array, self.theta)
                          
def crossval(model, inputs, targets, n_folds, center):
    
    n_samp = inputs.shape[0]
    numpy.random.seed(0)
    perm = numpy.random.permutation(n_samp)
    inputs = inputs[perm]
    targets = targets[perm]

    if len(targets.shape) == 1:
        targets = targets[:,None]
    test_error = numpy.zeros(targets.shape[1])
    train_error = numpy.zeros(targets.shape[1])
    weights = None

    for f in xrange(n_folds):

        ind_a = numpy.round(n_samp*f / float(n_folds)).astype(int)
        ind_b = numpy.round(n_samp*(f+1) / float(n_folds)).astype(int)
        mask = numpy.array([True]*n_samp)
        mask[ind_a:ind_b] = False
        not_mask = numpy.negative(mask)
        
        m = model(inputs[mask], targets[mask], center = center)
        train_error += m.test_prediction(inputs[mask], targets[mask])
        test_error += m.test_prediction(inputs[not_mask], targets[not_mask])
        weights = m.theta if weights is None else weights + m.theta

    return test_error / float(n_folds), train_error / float(n_folds), weights / float(n_folds)


def test_models(
    data_dir = 'data/satellite/processed/',
    target_channel = 'swd_sfc',
    n_folds = 10,
    size = 9,
    n_frames = 1,
    delta_time = 1., # in hours
    n_channels = 3,
    center = True):

    paths = glob.glob(data_dir + 'goes-insolation.dt-%0.1f-nf-%i.nc-%i.ws-%i.str-*.dens-*.nsamp-*.pickle.gz' % 
            (delta_time, n_frames, n_channels, size))

    assert len(paths) > 0

    print 'matching files: ', paths
    print 'using first: ', paths[0]

    with openz(paths[0]) as data_file:
        dataset = pickle.load(data_file)
    
    input_names = dataset['input-names']
    target_index = input_names.index(target_channel)
    targets = dataset['target'][:,target_index]
    inputs  = dataset['input']
    mn,std = dataset['stats'][target_index,:]
    
    n_samples, nf, nc, n_dims = inputs.shape
    assert nf == n_frames
    assert nc == n_channels
    assert size**2 == n_dims

    ar_test_error, ar_train_error, ar_weight_avg = crossval(ARmodel, inputs, targets, n_folds, center)
    np_test_error, np_train_error, _ = crossval(NeutralPredictor, inputs, targets, n_folds, center)

    ar_test_error = ar_test_error * std
    ar_train_error = ar_train_error * std
    np_test_error = np_test_error * std
    np_train_error = np_train_error * std

    return ar_test_error, ar_train_error, np_test_error, np_train_error

def performance_tests(
                     def_size = 11,
                     def_n_frames = 1,
                     def_delta_time = 1.,
                     def_n_channels = 3,
                     sizes = [3,5,7],
                     num_frames = [2,3,4],
                     delta_times = [3.,6.,12.]
                     ):

    df = pandas.DataFrame(columns = ['ar-test-error', 
                                     'ar-train-error', 
                                     'np-test-error', 
                                     'np-train-error',
                                     'window-size',
                                     'n-frames',
                                     'delta-time',
                                     'n-channels'])

    def run_experiment(size, n_frames, delta_time, n_channels):
        ar_test, ar_train, np_test, np_train = test_models(
                                                          size = size,
                                                          n_frames = n_frames,
                                                          delta_time = delta_time,
                                                          n_channels = n_channels
                                                          )
        df.append({'ar-test-error' : ar_test,
                   'ar-train-error': ar_train,
                   'np-test-error' : np_test,
                   'np-train-error': np_train,
                   'window-size' : size,
                   'n-frames' : n_frames,
                   'delta-time' : delta_time,
                   'n-channels' : n_channels}, ignore_index = True)


    # vary window size
    for size in sizes:
        run_experiment(
                      size = size, 
                      n_frames = def_n_frames, 
                      delta_time = def_delta_time, 
                      n_channels = def_n_channels
                      )
                                                          
    # vary number of frames
    for nf in num_frames:
        run_experiment(
                      size = def_size, 
                      n_frames = nf, 
                      delta_time = def_delta_time, 
                      n_channels = def_n_channels
                      )

    # vary forecast interval with two different window sizes
    for dt in delta_times:
        run_experiment(
                      size = def_size, 
                      n_frames = def_n_frames, 
                      delta_time = dt, 
                      n_channels = def_n_channels
                      )
    
        # change window size to larger than default
        run_experiment(
                      size = 19, 
                      n_frames = def_n_frames, 
                      delta_time = dt, 
                      n_channels = def_n_channels
                      )
    timestamp = str(datetime.datetime.now().replace(microsecond=0)).replace(' ','|')
    df.to_csv('data/satellite/output/ar_performance_tests%s.csv' % timestamp)

def plot_performance(data_dir = 'data/satellite/output/'):
    
    paths = glob.glob(data_dir + 'ar_performance_tests*.csv')
    sorted(paths)
    df = pandas.read_csv(paths[-1]) # read most recent results
    
    # performance v window size
    plt.subplot(311,1)
    gb = df.groupby(['n-frames','delta-time','n-channels'])
    for indx, group in gb:
        plt.plot(df['window-size'], df['ar-test-error'], label = str(indx)+' AR test') 
        plt.plot(df['window-size'], df['ar-train-error'], label = str(indx)+' AR train')
        plt.plot(df['window-size'], df['np-test-error'], label = str(indx)+' NP test')
        plt.plot(df['window-size'], df['np-train-error'], label = str(indx)+' NP train')
    plt.legend()
    
    # performance v number of frames
    plt.subplot(311,2)
    gb = df.groupby(['window-size','delta-time','n-channels'])
    for indx, group in gb:
        plt.plot(df['n-frames'], df['ar-test-error'], label = str(indx)+' AR test') 
        plt.plot(df['n-frames'], df['ar-train-error'], label = str(indx)+' AR train') 
        plt.plot(df['n-frames'], df['np-test-error'], label = str(indx)+' NP test') 
        plt.plot(df['n-frames'], df['np-train-error'], label = str(indx)+' NP train') 
    plt.legend()
    
    # performance v forecast interval (delta time)
    plt.subplot(311,3)
    gb = df.groupby(['window-size','n-frames','n-channels'])
    for indx, group in gb:
        plt.plot(df['delta-time'], df['ar-test-error'], label = str(indx)+' AR test') 
        plt.plot(df['delta-time'], df['ar-train-error'], label = str(indx)+' AR train')
        plt.plot(df['delta-time'], df['np-test-error'], label = str(indx)+' NP test')
        plt.plot(df['delta-time'], df['np-train-error'], label = str(indx)+' NP train')
    plt.legend()

    plt.savefig(paths[-1][:-4] + '.png')


def plot_weights(weight, n_frames, n_channels, n_dims, size, names, center = True):
    
    wt_img = weight[:-1,-1] if center else weight[:,-1]
    wt_img = numpy.reshape(wt_img, (n_frames, n_channels, n_dims))

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
            grid[ind].title.set_text(names[i])
                
    grid.cbar_axes[0].colorbar(im) #, format='%.2f')
    plt.savefig('ar_weights.nf-%i.nc-%i.ws-%i.png' % (n_frames, n_channels, size))

if __name__ == '__main__':
    plac.call(performance_tests)
