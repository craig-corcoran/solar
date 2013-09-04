import re
import copy
import datetime
import numpy
import glob
import plac
import pandas
import cProfile
import cPickle as pickle
import multiprocessing as mp
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
from solar.util import openz

# split experiments into several phases
# reparallelize
# profile performance
# numpy installation optimal?

class NeutralPredictor(object):

    def __init__(self, input_array, target_array, center = None, target_index = -1):
        
        self.n_dims = input_array.shape[-1]
        self.theta = numpy.zeros(input_array.shape[1:])
        self.theta[-1,:,numpy.ceil(self.n_dims/2.)] = 1.
        self.target_index = target_index
        
    def make_prediction(self, input_array):
        
        assert input_array.ndim == 4 # n_samples, n_frames, n_channels, n_dims
        return input_array[:,-1,self.target_index,numpy.ceil(self.n_dims/2.)][:,None] 

    def test_prediction(self, input_array, target_array):
        
        return numpy.sqrt(numpy.mean(
                  (self.make_prediction(input_array)-target_array)**2,
                  axis=0)) 

class ARmodel(NeutralPredictor):
    ''' Autoregressive model, takes input and target arrays and computes the 
    least squares weight coefficients '''
    def __init__(self, input_array, target_array, l2reg = 0., center = True, target_index = -1):
        
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

def test_model(model, train_inputs, train_targets, test_inputs, test_targets, center, target_index):
    m = model(train_inputs, train_targets, center, target_index)
    train_error = m.test_prediction(train_inputs, train_targets)
    test_error = m.test_prediction(test_inputs, test_targets)
    return test_error, train_error, m.theta
                          
def crossval(model, inputs, targets, n_folds, center, target_index):
    
    n_samp = inputs.shape[0]
    numpy.random.seed(0)
    perm = numpy.random.permutation(n_samp)
    inputs = inputs[perm]
    targets = targets[perm]

    if len(targets.shape) == 1:
        targets = targets[:,None]
    n_targets = targets.shape[1]
    pool = mp.Pool(min(mp.cpu_count(), n_folds))
    
    class Errors(object):
        
        def __init__(self):
            self.test_error = numpy.zeros(n_targets)
            self.train_error = numpy.zeros(n_targets)
            self.weights = None
        
        def accumulate_errors(self, inp):
            #print 'accumulate error input: ', inp
            ts_error, tr_error, wts = inp
            self.test_error += ts_error
            self.train_error += tr_error
            self.weights = wts if self.weights is None else self.weights + wts
    
    errors = Errors()
    for f in xrange(n_folds):
        
        print 'fold number: ', f    
        ind_a = numpy.round(n_samp*f / float(n_folds)).astype(int)
        ind_b = numpy.round(n_samp*(f+1) / float(   n_folds)).astype(int)
        mask = numpy.array([True]*n_samp)
        mask[ind_a:ind_b] = False
        not_mask = numpy.negative(mask)
        
        pool.apply_async(test_model, 
                         args = [model, 
                                inputs[mask], targets[mask], 
                                inputs[not_mask], targets[not_mask], 
                                center, target_index],
                         callback = errors.accumulate_errors)
    pool.close()
    pool.join()

    assert not (errors.test_error == 0).all()
    assert errors.weights is not None

    return errors.test_error / float(n_folds), errors.train_error / float(n_folds), errors.weights / float(n_folds)

def unpack_dataset(path):
        with openz(path) as data_file:
            ds = pickle.load(data_file)
        print 'unpacked file: ', path
        return ds  

def test_models(
    data_dir = 'data/satellite/processed/',
    target_channel = 'swd_sfc',
    n_folds = 10,
    size = 9,
    n_frames = 1,
    delta_time = 1., # in hours
    n_channels = 3,
    center = True):
    
    paths = glob.glob(data_dir + 'goes-insolation.dt-%0.1f.nf-%i.nc-%i.ws-%i.str-*.dens-*.nsamp-*.0.pickle.gz' % 
            (delta_time, n_frames, n_channels, size))

    assert len(paths) > 0

    print 'matching files: ', paths
    print 'using first: ', paths[0]
    
    # find all files from the same batch
    path = paths[0]
    part = path[:-11] 
    paths = glob.glob(part+'*.pickle.gz')

    par_grp = re.match(
        '.*goes.*\.nf-(\d+)\.nc-(\d+)\.ws-(\d+).*\.nsamp-(\d+)\.\d+\.pickle\.gz',
        path).groups()
    n_frames, n_channels, wind_size, n_samples = map(int, par_grp)
    n_wind_cells = wind_size**2
    
    class Dataset(object):
        
        def __init__(self, n_samples, n_frames, n_channels, n_wind_cells):
            self.data = {}
            self.n_frames = n_frames
            self.n_channels = n_channels
            self.inputs = numpy.zeros((n_samples, n_frames, n_channels, n_wind_cells)) * numpy.nan
            self.targets = numpy.zeros((n_samples, n_channels)) * numpy.nan
            self.timestamps = numpy.zeros((n_samples, n_frames+1), dtype = object) * numpy.nan        
            self.i = 0

        def accumulate(self, ds):
            if self.i == 0:
                print 'first accumulate'
                self.data = copy.deepcopy(ds)
                self.data['windows'] = self.inputs
                self.data['targets'] = self.targets
                self.data['timestamps'] = self.timestamps
            else:
                print 'not first accumulate'
                # assert parameters are consistent across batch
                assert self.data['input-names'] == ds['input-names']
                #assert (self.data['norm-stats'] == ds['norm-stats']).all()
                assert self.data['n-samples'] == ds['n-samples']
                assert self.data['n-frames'] == ds['n-frames'] == self.n_frames
                assert self.data['n-channels'] == ds['n-channels'] == self.n_channels
                #assert self.data['sample-strid'] == ds['sample-stride'] 
                #assert self.data['dens-thresh'] == ds['dens-thresh'] 
                #assert self.data['lon-range'] == ds['lat-range']
                #assert self.data['dlat'] == ds['dlat']
                #assert self.data['dlon'] == ds['dlon']
                #assert self.data['dt'] == ds['dt']
                #assert self.data['window-shape'] == ds['window-shape']
                #assert self.data['normalized'] == ds['normalized']
        
            print 'current index: ', self.i
            print ds['windows'].shape
            ns = ds['windows'].shape[0]
            self.inputs[self.i:self.i+ns] = ds['windows']
            self.targets[self.i:self.i+ns] = ds['targets']
            self.timestamps[self.i:self.i+ns] = ds['timestamps']
            self.i += ns        
            print 'new index: ', self.i     

    dataset = Dataset(n_samples, n_frames, n_channels, n_wind_cells)
        
    pool = mp.Pool(min(mp.cpu_count(), len(paths)))
    for p in paths:
        print 'opening file: ', p
        pool.apply_async(unpack_dataset, 
                        args = [p], 
                        callback=dataset.accumulate)
    
    pool.close()
    pool.join()
    
    while (numpy.isnan(dataset.inputs)).any():
        print 'dataset not filled'

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
    
    print 'performing crossval for AR'
    ar_test_error, ar_train_error, ar_weight_avg = crossval(ARmodel, inputs, targets, n_folds, center, target_index)
    print 'performing crossval for neutral predictor'
    np_test_error, np_train_error, _ = crossval(NeutralPredictor, inputs, targets, n_folds, center, target_index)

    ar_test_error = ar_test_error * std
    ar_train_error = ar_train_error * std
    np_test_error = np_test_error * std
    np_train_error = np_train_error * std

    print 'finished crossval'

    return ar_test_error, ar_train_error, np_test_error, np_train_error

def run_experiment(size, n_frames, delta_time, n_channels):
        (ar_test, ar_train,
        np_test, np_train) = test_models(
                                         size = size,
                                         n_frames = n_frames,
                                         delta_time = delta_time,
                                         n_channels = n_channels
                                         )

        return {'ar-test-error' : ar_test[0],
                'ar-train-error': ar_train[0],
                'np-test-error' : np_test[0],
                'np-train-error': np_train[0],
                'window-size' : size,
                'n-frames' : n_frames,
                'delta-time' : delta_time,
                'n-channels' : n_channels}

def performance_tests(
                     def_size = 11,
                     def_n_frames = 1,
                     def_delta_time = 1.,
                     def_n_channels = 3,
                     to_vary = 'sizes', # 'num-frames', delta-times'
                     params = [3,5,7,9,11,15,19], # [2,3,4], [3.,6.,24.]
                     ):

    class DataTable(object):
        
        def __init__(self):
            self.data = pandas.DataFrame(columns = [
                                         'ar-test-error', 
                                         'ar-train-error', 
                                         'np-test-error', 
                                         'np-train-error',
                                         'window-size',
                                         'n-frames',
                                         'delta-time',
                                         'n-channels'])

        def update_frame(self, dic):            
            self.data.append(dic, ignore_index = True)
    
    pool = mp.Pool(min(mp.cpu_count(), len(params)))

    data_table = DataTable()
    for p in params:
        
        if to_vary is 'sizes':
            inputs = [p, def_n_frames, def_delta_time, def_n_channels]

        elif to_vary is 'num-frames':
            inputs = [def_size, p, def_delta_time, def_n_channels]

        elif to_vary is 'delta-times':
            inputs = [def_size, def_n_frames, p, def_n_channels]
        else: assert False
        
        #pool.apply_async(run_experiment, args=inputs, callback = data_table.update_frame)
        data_table.update_frame(run_experiment(*inputs))

    pool.close()
    pool.join()

    timestamp = str(datetime.datetime.now().replace(microsecond=0)).replace(' ','|')
    data_table.data.to_csv('data/satellite/output/ar_performance_tests.%s.%s.csv' % (to_vary, timestamp))

def plot_performance(data_dir = 'data/satellite/output/'):
    
    paths = glob.glob(data_dir + 'ar_performance_tests*.csv')
    sorted(paths)
    df = pandas.read_csv(paths[-1]) # read most recent results
    
    # performance v window size
    #plt.subplot(311,1)
    gb = df.groupby(['n-frames','delta-time','n-channels'])
    for indx, group in gb:
        print indx
        print group
        #print 'ar errors: ', numpy.array(group['ar-test-error'].values[0][1:-1].split(']['), dtype = float)    
        plt.plot(group['window-size'], group['ar-test-error'], label = str(indx)+' AR test')
        plt.plot(group['window-size'], group['ar-train-error'], label = str(indx)+' AR train')

        print group['np-test-error']
        print group['ar-test-error']
        print group['ar-train-error']
        plt.plot(group['window-size'], group['np-test-error'], label = str(indx)+' NP test')
        #plt.plot(group['window-size'], group['np-train-error'], label = str(indx)+' NP train')
    plt.legend()
    
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
    #plac.call(plot_performance)
    #cProfile.run('performance_tests')
    plac.call(performance_tests)
