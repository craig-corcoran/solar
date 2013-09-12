import re
import copy
import datetime
import numpy
import glob
import plac
import pandas
import cPickle as pickle
import multiprocessing as mp
from solar.util import openz
from solar.learning.models import NeutralPredictor, ARmodel

def test_model(model, inputs, targets, mask, center, target_index):
    
    ''' gets the model errors and weights given inputs, targets, and mask array
    separating test from training errors. '''

    print 'testing model: ', model
    not_mask = numpy.negative(mask)
    m = model(inputs[mask], targets[mask], center, target_index)
    print 'model learned'
    train_error = m.test_prediction(inputs[mask], targets[mask])
    print 'test error measured'
    test_error = m.test_prediction(inputs[not_mask], targets[not_mask])
    print 'returning from test model'
    return test_error, train_error, m.theta
                          
def crossval(model, inputs, targets, n_folds, center, target_index):

    ''' performs n_folds model tests, returning the mean and std errors and the 
    average weights in a dictionary.

    model : model object (ex. ARmodel)
    inputs : input windows
    targets : desired outputs
    n_folds : number of folds for crossval averaging
    center : boolean, should we add a constant column to the features 
             (effectively centering the data)?
    target_index : integer specifying which of the channels is the one we wish 
             to predict (usually -1)
    '''
    
    n_samp = inputs.shape[0]
    numpy.random.seed(0)
    perm = numpy.random.permutation(n_samp)
    inputs = inputs[perm]
    targets = targets[perm]

    if len(targets.shape) == 1:
        targets = targets[:,None] # we need targets to be a 2D array
    n_targets = targets.shape[1]
    
    
    class Errors(object):
        
        ''' Error object for accumulating test and training errors as well as 
        the average weights '''

        def __init__(self):
            self.test_error = numpy.zeros((n_folds, n_targets))
            self.train_error = numpy.zeros((n_folds, n_targets))
            self.weights = None
            self.ind = 0
        
        def accumulate_errors(self, inp):
            
            ts_error, tr_error, wts = inp 
            self.test_error[self.ind] = ts_error
            self.train_error[self.ind] = tr_error
            self.weights = wts if self.weights is None else self.weights + wts
            self.ind += 1
    
    errors = Errors()
    pool = mp.Pool(min(mp.cpu_count(), n_folds))    

    for f in xrange(n_folds):
        
        print 'fold number: ', f    
        ind_a = numpy.round(n_samp*f / float(n_folds)).astype(int)
        ind_b = numpy.round(n_samp*(f+1) / float(   n_folds)).astype(int)
        mask = numpy.array([True]*n_samp)
        mask[ind_a:ind_b] = False
        
        pool.apply_async(test_model, 
                    args = [model, inputs, targets, mask, center, target_index], 
                    callback = errors.accumulate_errors)
    pool.close()
    pool.join()

    assert not (errors.test_error == 0).all()
    assert errors.weights is not None
    
    # XXX not compatible with n_targets > 1
    return {'test-error': numpy.mean(errors.test_error), 
           'train-error': numpy.mean(errors.train_error),
           'test-std': numpy.std(errors.test_error), 
           'train-std': numpy.std(errors.train_error),
           'avg-weights': errors.weights / float(n_folds)}

class Dataset(object):

    ''' Dataset object used for accumulating training inputs and outputs 
    from several files '''
    
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
            self.data = copy.deepcopy(ds)
            self.data['windows'] = self.inputs
            self.data['targets'] = self.targets
            self.data['timestamps'] = self.timestamps
        else:
            # assert parameters are consistent across batch
            assert self.data['input-names'] == ds['input-names']
            assert self.data['n-samples'] == ds['n-samples']
            assert self.data['n-frames'] == ds['n-frames'] == self.n_frames
            assert self.data['n-channels'] == ds['n-channels'] == self.n_channels
            assert self.data['dens-thresh'] == ds['dens-thresh'] 
            assert self.data['dlat'] == ds['dlat']
            assert self.data['dlon'] == ds['dlon']
            assert self.data['dt'] == ds['dt']
            assert self.data['window-shape'] == ds['window-shape']
            assert self.data['normalized'] == ds['normalized']
    
        ns = ds['windows'].shape[0]
        self.inputs[self.i:self.i+ns] = ds['windows']
        self.targets[self.i:self.i+ns] = ds['targets']
        self.timestamps[self.i:self.i+ns] = ds['timestamps']
        self.i += ns


def unpack_dataset(path):
    ''' unpickle the object at the specified path, possibly gzipped '''
    with openz(path) as data_file:
        return pickle.load(data_file)  

def get_models_performances(
    data_dir = 'data/satellite/processed/',
    target_channel = 'swd_sfc',
    n_folds = 10,
    size = 9,
    n_frames = 1,
    delta_time = 1., 
    n_channels = 3,
    center = True, 
    gzip = False):

    ''' unpacks all processed (pickled) data from the data_dir, then evaluates
    the performance of a neutral predictor and a linear AR model using n_folds
    fold crossvalidation 
    
    data_dir : directory to read data files from, will use all files in this 
               directory with matching parameters

    target_channel : string, name of the channel we want to predict 
    n_folds : int, number of folds used for crossvalidation
    size : int, size of square window of inputs, each input channel is size x size in dimension
    n_frames : int, number of time frames into the past. n_frames = 1 corresponds to an AR1 model
    delta_time : float, prediction interval in hours, also time difference between frames (if > 1)
    n_channels : int, number of channels in the input signal
    center : boolean, should we use a constant feature (effectively centering the data)?
    '''
    
    file_string = 'goes-insolation.dt-%0.1f.nf-%i.nc-%i.ws-%i.str-*.dens-*.nsamp-*.0.pickle'
    if gzip: file_string += '.gz'
    paths = glob.glob(data_dir + file_string % 
                        (delta_time, n_frames, n_channels, size)) 

    assert len(paths) > 0
    sorted(paths)
    print 'matching files: ', paths
    print 'using: ', paths[0] 
    
    # find all files from the same batch
    path = paths[0]
    part = path[:-11] if gzip else path[:-9]
    paths = glob.glob(part+'*.pickle.gz') if gzip else glob.glob(part+'*.pickle')
    
    print 'number of files being used: ', len(paths)
    
    match_string = '.*goes.*\.nf-(\d+)\.nc-(\d+)\.ws-(\d+).*\.nsamp-(\d+)\.\d+\.pickle'
    if gzip: match_string += '.gz'
    par_grp = re.match(match_string, path).groups()
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
    
    print 'performing crossval for AR'
    ar_results = crossval(ARmodel, inputs, targets, n_folds, center, target_index)
    print 'performing crossval for neutral predictor'
    np_results = crossval(NeutralPredictor, inputs, targets, n_folds, center, target_index)
    
    for key in ar_results:
        if key is not 'avg-weights':
            ar_results[key] = ar_results[key] * std
            np_results[key] = np_results[key] * std

    print 'finished crossval'

    return ar_results, np_results 

def run_experiment(size, n_frames, delta_time, n_channels):

    ''' runs performance experiments then unpacks results into a dictionary '''

    ar_results, np_results = get_models_performances(
                                        size = size,
                                        n_frames = n_frames,
                                        delta_time = delta_time,
                                        n_channels = n_channels
                                        )

    return {'ar-test-error' : ar_results['test-error'], 
            'ar-train-error': ar_results['train-error'],
            'np-test-error' : np_results['test-error'],
            'np-train-error': np_results['train-error'],
            'ar-test-std' : ar_results['test-std'], 
            'ar-train-std': ar_results['train-std'],
            'np-test-std' : np_results['test-std'],
            'np-train-std': np_results['train-std'],
            'window-size' : size,
            'n-frames' : n_frames,
            'delta-time' : delta_time,
            'n-channels' : n_channels}

# XXX input, output directory?
@plac.annotations(
def_size = ('default window size, used if to_vary is not sizes', 'option', None, int),
def_n_frames = ('default number of frames into the past used, used if to_vary is not num-frames', 'option', None, int),
def_delta_time = ('default forecast interval and time between frames in hours, used if to_vart is not delta-times', 'option', None, float),
def_n_channels = ('default number of channels, currently no to_vary for channels', 'option', None, int),
to_vary = ('quantity that is being varied for this experiment', 'option', ['sizes', 'num-frames', 'delta-times'], str),
params = ('values used for to_vary quantity, usually a list of ints', 'option', None, None))
def performance_experiment(
                     def_size = 11,
                     def_n_frames = 1,
                     def_delta_time = 3,
                     def_n_channels = 3,
                     to_vary = 'sizes', # 'num-frames', delta-times'
                     params = [3,5,7,9,11,15,19]
                     ):

    ''' script for running an experiment using AR and neutral prediction models
    varying the to_vary parameter with parameter values params. writes the 
    output to a csv file '''

    class DataTable(object):
        
        ''' small wrapper for a pandas dataframe with an update method for 
        appending new rows to the dataframe given a dictionary to append'''

        def __init__(self):
            self.data = pandas.DataFrame(columns = [
                                         'ar-test-error', 
                                         'ar-train-error', 
                                         'np-test-error', 
                                         'np-train-error',
                                         'ar-test-std', 
                                         'ar-train-std', 
                                         'np-test-std', 
                                         'np-train-std',
                                         'window-size',
                                         'n-frames',
                                         'delta-time',
                                         'n-channels'])

        def update_frame(self, dic):            
            self.data = self.data.append(dic, ignore_index = True)
    
    data_table = DataTable()
    
    #pool = mp.Pool(min(mp.cpu_count(), len(params)))
    for p in params:
        print to_vary
        print p
        if to_vary == 'sizes':
            inputs = [p, def_n_frames, def_delta_time, def_n_channels]

        elif to_vary == 'num-frames':
            inputs = [def_size, p, def_delta_time, def_n_channels]

        elif to_vary == 'delta-times':
            inputs = [def_size, def_n_frames, p, def_n_channels]
        else:
            print to_vary
            assert False
        
        # XXX haven't been able to parallelize this
        #pool.apply_async(run_experiment, args=inputs, callback = data_table.update_frame)
        data_table.update_frame(apply(run_experiment, inputs))
        
    #pool.close()
    #pool.join()        
        
    dt = inputs[2]    
    timestamp = str(datetime.datetime.now().replace(microsecond=0)).replace(' ','|')
    data_table.data.to_csv('data/satellite/output/ar_performance_tests.%s.dt-%i.%s.csv' % (to_vary, dt, timestamp))


if __name__ == '__main__':
    plac.call(performance_experiment)
    
