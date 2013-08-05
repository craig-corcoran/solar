import numpy
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
    
    def __init__(self, input_array, target_array, l2reg = 0.4, center = True):
        
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


def main(
    data_path = 'solar/data/processed/goes-insolation.nf-1.nc-3.ws-9.str-2.dth-0.6.pickle.gz',
    n_folds = 10,
    center = True):

    with openz(data_path) as data_file:
        dataset = pickle.load(data_file)
    
    targets = dataset['target']
    inputs  = dataset['input']
    mn,std = dataset['stats'][-1,:]
    
    n_samples, n_frames, n_channels, n_dims = inputs.shape
    side = numpy.sqrt(n_dims).astype(int)

    ar_test_error, ar_train_error, ar_weight_avg = crossval(ARmodel, inputs, targets, n_folds, center)
    np_test_error, np_train_error,_ = crossval(NeutralPredictor, inputs, targets, n_folds, center)
    
    # XXX store target labels
    if center:
        print 'learned constant bias: ', ar_weight_avg[-1,-1]

    print 'number of frames: ', n_samples
    print 'number of channels: ', n_channels
    print 'number of spatial dimensions: ', n_dims
    print 'number of parameters: ', n_channels * n_dims * n_frames
    print 'number of samples: ', n_samples, '\n'
    print '%i fold crossval' %n_folds
    print 'AR test error: ', ar_test_error * std + mn
    print 'AR train error: ', ar_train_error * std + mn
    print 'Neutral test error: ', np_test_error * std + mn
    print 'Neutral train error: ', np_train_error * std + mn, '\n'
    print '% improvement in test error: ', 100.*(np_test_error-ar_test_error) / np_test_error
    print '% improvement in train error: ', 100.*(np_train_error-ar_train_error) / np_train_error
    
    wt_img = ar_weight_avg[:-1,-1] if center else ar_weight_avg[:,-1]
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
                                 numpy.reshape(wt_img[t,i,:], (side,side)), 
                                 vmin = vmin, vmax = vmax,
                                 origin = 'lower',
                                 interpolation = 'nearest'
                                 )
            grid[ind].title.set_text(dataset['names'][i])
                
    grid.cbar_axes[0].colorbar(im) #, format='%.2f')
    plt.savefig('ar_weights.nf-%i.nc-%i.ws-%i.png' % (n_frames, n_channels, side))


if __name__ == '__main__':
    main()
