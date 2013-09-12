import numpy


class NeutralPredictor(object):

    ''' Neutral predictor simply predicts the value at the center pixel of the 
    target_index channel at time t for time t+1 '''

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
    least squares weight coefficients. Note ARmodel is inheriting the test 
    prediction method from the neutral predictor (NP) and also doesn't use the 
    target_index, but needs it to have the same interface at the NP.'''

    def __init__(self, input_array, target_array, l2reg = 0., center = True, target_index = -1):
        
        print 'initializing AR model'

        self.n_samples = input_array.shape[0]

        if len(input_array.shape) > 2:
            input_array = self._flatten_samples(input_array)
        
        self.center = center
        if self.center:
            input_array = numpy.hstack([input_array, numpy.ones((input_array.shape[0],1))])
        
        #self.theta = numpy.linalg.lstsq(input_array, target_array)[0]

        print 'constructing reg and covariance matrices'
        self.l2reg = l2reg
        reg_matrix = self.l2reg * numpy.eye(input_array.shape[1])
        if center:
            reg_matrix[-1,-1] = 0.
        C = (1./self.n_samples) * numpy.dot(input_array.T, input_array) + reg_matrix
        b = (1./self.n_samples) * numpy.dot(input_array.T, target_array)
        print 'solving for optimal linear weights'
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

