import plac
import glob
import pandas
import numpy
from matplotlib import pyplot

class MixtureModel(object):
    
    def __init__(self, A, target, mask, l2reg = 0., center = True):
        
        self.centered = center
        if center: # append constant column
            A = numpy.hstack([A, numpy.ones((A.shape[0],1))])

        not_mask = numpy.array(map(lambda x: not x, mask))
        self.target = {'train': target[mask], 'test': target[not_mask]} 
        self.A  = {'train': A[mask,:], 'test': A[not_mask,:]} 

        C = numpy.dot(A['train'].T, A['train']) + l2reg * numpy.eye(A['train'].shape[1])
        b = numpy.dot(A['train'].T, meas['train'])
        self.w = numpy.linalg.solve(C, b)

    def model_errors(self):
        ''' returns dictionaries with the test and training RMSE and standard 
        error for each column in A, which assumes each column is a model trying 
        to predict the target values
        '''
        mn_err = {}; std_err = {}
        for dset in ['train', 'test']:
            err = (self.A[dset] - self.target[dset])**2
            if self.center:
                err = err[:, :-1]
            mn_err[dset] = numpy.sqrt(numpy.mean(err, axis = 0))
            std_err[dset] = numpy.sqrt(numpy.std(err, axis = 0)) / numpy.sqrt(len(self.target[dset]))
        return mn_err, std_err
    
    def mixture_errors(self):
        ''' returns dictionaries with the test and training RMSE and standard 
        error for the mixture using the learned weights w.
        '''
        mn_err = {}; std_err = {}
        for dset in ['train', 'test']:
            err = (numpy.dot(self.A[dset], self.w) - self.target[dset])**2
            mn_err[dset] = numpy.sqrt(numpy.mean(err, axis = 0))
            std_err[dset] = numpy.sqrt(numpy.std(err, axis = 0)) / numpy.sqrt(len(self.target[dset]))
        return mn_err, std_err
        

def main(
    path_root = 'data/prediction/',
    to_predict = 'Total', #'Diff',# 'DirNorm', #
    l2reg = 0., 
    hold_out = None, #(5,16),
    n_folds = 10,
    center = True,
    plot = True, 
    models = ['mix','hrrr','nam']):

    paths = glob.glob(path_root + '/*.csv')
    paths = sorted(paths)
    n_sites = len(paths)
    n_params = len(models) + (1 if center else 0)
    
    error = {}; std_err = {};
    for mod in models:
        error[mod] = numpy.zeros(n_sites) 
        std_err[mod] = numpy.zeos(n_sites)
    weights = numpy.zeros((n_sites, n_params))
    n_samples = numpy.zeros(n_sites)

    for i,p in enumerate(paths):
        
        df = pandas.read_csv(p)
        
        meas = df['irrMe%s' % to_predict]    
        hrrr = df['irrMoOp1_%s' % to_predict]    
        nam = df['irrMoOp2_%s' % to_predict]
        A = numpy.vstack([hrrr, nam]).T
        
        n_samples = len(meas)
            
            
        if (hold_out is None) & (n_folds > 0):
            s_err = dict(zip(models, numpy.zeros(len(models))))
            err = dict(zip(models, numpy.zeros(len(models))))
            wt = None
            for f in xrange(n_folds):
                
                #print 'fold:; ', f

                ind_a = numpy.round(n_samples*f / float(n_folds)).astype(int)
                ind_b = numpy.round(n_samples*(f+1) / float(n_folds)).astype(int)
                mask = numpy.array([True]*n_samples)
                mask[ind_a:ind_b] = False
                w, er, se = train_model(meas, hrrr, nam, mask, l2reg, center, abs_diff)
                wt = w if wt is None else wt+w
                for k in er.keys():  # sum over folds
                    err[k] += er[k]
                    s_err[k] += se[k]

                #print 'fold %i error: %f' % (f, er['mix'])
                #print 'fold %i std error: %f' % (f, se['mix'])
                #print 'weights: ', w

            wt = wt / float(n_folds)
            for k in er.keys():  # average over folds
                err[k] = err[k] / float(n_folds)
                s_err[k] = s_err[k] / (float(n_folds) * numpy.sqrt(n_folds))
                
        elif (n_folds is None) & (len(hold_out) == 2):
            # find the first data point after hold out day
            mon_ind = numpy.where(df['month']==hold_out[0])[0][0]
            day_ind = numpy.where(df['day']==hold_out[1])[0]
            ind = day_ind[numpy.where(day_ind > mon_ind)[0][0]]
            mask = numpy.array([True]*n_samples)
            mask[ind:] = False

            m = MixtureModel(A, meas, mask, l2reg, center)
            # XXX
            wt, err, s_err = train_model(meas, hrrr, nam, mask, l2reg, center, abs_diff)
        else: 
            print 'n_folds or hold_out date must be set and the other None'
            assert false
        
        weights['hrrr'][i] = wt[0]
        weights['nam'][i] = wt[1]
        weights['const'][i] = wt[-1]
        for k in error.keys():
            error[k][i] = err[k]
            std_err[k][i] = s_err[k]



        # get per-day error?
        
        print 'file: ', p
        #print 'learned weights: ', w
        #print 'training index: ', ind
        #print 'mixture error: ', error_mix[i]
        #print 'hrrr error: ', error_hrrr[i]
        #print 'nam error: ', error_nam[i]

    print 'average mixture error: ', numpy.mean(error['mix'])
    print 'average hrrr error: ', numpy.mean(error['hrrr'])
    print 'average nam error: ', numpy.mean(error['nam'])
    
    if plot:

        if to_predict is 'Diff':
            to_predict = 'Diffuse'
        elif to_predict is 'DirNorm':
            to_predict = 'Direct Normal'

        ind = numpy.arange(n_sites)
        
        # plot errors by site
        pyplot.clf()
        ax = pyplot.subplot(111)
        ax.bar(ind-0.2, error['mix'], width=0.2, color='g', align='center', yerr=std_err['mix'], ecolor='k')
        ax.bar(ind, error['hrrr'], width=0.2, color='b', align='center', yerr=std_err['hrrr'], ecolor='k')
        ax.bar(ind+0.2, error['nam'], width=0.2, color='r', align='center', yerr=std_err['nam'], ecolor='k')
        pyplot.legend(['LR Mixture', 'HRRR', 'NAM'], fontsize = 20, loc=2)
        site_codes = ['bon','tbl','dra','fpk','gwn','psu','sxf','abq','bis','hxn','msn','sea','slc','ste']
        pyplot.xticks(ind, site_codes, size = 25, fontweight='bold')
        pyplot.ylabel('RMSE $(W/m^2)$', size = 20, fontweight='bold')
        pyplot.suptitle('%s Irradiance %s Error by Site' % 
                (to_predict, 'Forecast' if n_folds is None else 'Crossval'), 
                fontsize = 30, fontweight='bold')
        ax.autoscale(tight=False)
        pyplot.savefig('plots/site_error_%s-%s.pdf' % (to_predict.replace(' ',''),'Forecast' if n_folds is None else 'Crossval'))

        # plot learned weights by site
        pyplot.clf()
        ax = pyplot.subplot(211)
        ax.bar(ind-0.1, weights['hrrr'], width=0.2, color='b', align='center')
        ax.bar(ind+0.1, weights['nam'], width=0.2, color='r', align='center')
        pyplot.legend(['HRRR', 'NAM'], fontsize = 20, loc=2)
        pyplot.xticks(ind, ['']*14, size = 25)
        pyplot.ylabel('Regression Weight', size = 20, fontweight='bold')
        pyplot.suptitle('%s Linear Regression Weights by Site - %s' % 
                    ('Forecast' if n_folds is None else 'Crossval', to_predict),
                    fontsize = 30, fontweight='bold')
        ax.autoscale(tight=False)

        ax = pyplot.subplot(212)
        ax.bar(ind, weights['const'], width=0.2, color='k', align='center')
        pyplot.legend(['Const'], fontsize = 20,  loc=0)
        pyplot.xticks(ind, site_codes, size = 25, fontweight='bold')
        pyplot.ylabel('Irradiance $(W/m^2)$', size = 20, fontweight='bold')
        ax.autoscale(tight=False)
        pyplot.savefig('plots/site_weights_%s-%s.pdf' % \
            (to_predict.replace(' ',''),'Forecast' if n_folds is None else 'Crossval'))

def train_model(meas, hrrr, nam, mask, l2reg = 0., center = True, abs_diff = False):

    A = numpy.vstack([hrrr, nam]).T
    if abs_diff:
        A = numpy.hstack([A, numpy.abs(nam - hrrr)[:,None]])
    if center:
        A = numpy.hstack([A, numpy.ones((A.shape[0],1))])
    not_mask = numpy.array(map(lambda x: not x, mask))
    meas = {'train': meas[mask], 'test': meas[not_mask]} 
    hrrr = {'train': hrrr[mask], 'test': hrrr[not_mask]} # need hrrr/nam training?
    nam = {'train': nam[mask], 'test': nam[not_mask]} 
    A  = {'train': A[mask,:], 'test': A[not_mask,:]} 

    C = numpy.dot(A['train'].T, A['train']) + l2reg * numpy.eye(A['train'].shape[1])
    b = numpy.dot(A['train'].T, meas['train'])
    w = numpy.linalg.solve(C, b)
   
    sq_err = {'mix': (meas['test'] - numpy.dot(A['test'],w))**2,
              'hrrr': (meas['test'] - hrrr['test'])**2,
              'nam': (meas['test'] - nam['test'])**2}
    
    error = {}; std_err = {};
    for mod in ['mix', 'hrrr', 'nam']:
        error[mod] = numpy.sqrt(numpy.mean(sq_err[mod]))
        std_err[mod] = numpy.sqrt(numpy.std(sq_err[mod])) / numpy.sqrt(len(sq_err[mod]))

    return w, error, std_err    

def split_mixture(
    threshold = [20.],
    path_root = 'prediction',
    l2reg = 20., 
    hold_out = (5,16),
    center = True,
    n_parts = 2,
    ):

    assert len(threshold) == (n_parts - 1) 
    assert threshold == sorted(threshold)
    
    paths = glob.glob(path_root + '/*.csv')
    n_sites = len(paths)
    
    error_total = {'mix' : numpy.zeros(n_sites),
                   'hrrr': numpy.zeros(n_sites),
                   'nam' : numpy.zeros(n_sites)}

    error_part = {'mix' : numpy.zeros((n_parts, n_sites)),
                  'hrrr': numpy.zeros((n_parts, n_sites)),
                  'nam' : numpy.zeros((n_parts, n_sites))}

    for i,p in enumerate(paths):
        
        df = pandas.read_csv(p)
        
        meas = numpy.array(df['irrMeTotal'])    
        hrrr = numpy.array(df['irrMoOp1_Total'])    
        nam = numpy.array(df['irrMoOp2_Total'])
        diff = numpy.abs(nam-hrrr)

        # find the first data point after hold out day
        may_ind = numpy.where(df['month']==hold_out[0])[0][0]
        day_ind = numpy.where(df['day']==hold_out[1])[0]
        ind = day_ind[numpy.where(day_ind > may_ind)[0][0]]
        
        # split data into training and test
        meas, hrrr, nam, diff = map(lambda x: {'train':x[:ind], 'test':x[ind:]}, 
                                 [meas,hrrr,nam,diff])
        
        A = {}
        # partition dataset by threshold values for diff bt hrrr and nam
        for dset in ['train', 'test']:
            
            meas_split = [numpy.array([], dtype=numpy.float)]*n_parts
            hrrr_split = [numpy.array([], dtype=numpy.float)]*n_parts
            nam_split  = [numpy.array([], dtype=numpy.float)]*n_parts
            diff_split = [numpy.array([], dtype=numpy.float)]*n_parts
            for j in xrange(n_parts-1):

                meas_split[j] = meas[dset][diff[dset] < threshold[j]]
                hrrr_split[j] = hrrr[dset][diff[dset] < threshold[j]]
                nam_split[j]  =  nam[dset][diff[dset] < threshold[j]]
                diff_split[j] = diff[dset][diff[dset] < threshold[j]]
            
                if j == (n_parts - 2): # if on the last partition
                    meas_split[j+1] = meas[dset][diff[dset] >= threshold[j]]
                    hrrr_split[j+1] = hrrr[dset][diff[dset] >= threshold[j]]
                    nam_split[j+1]  = nam[dset][diff[dset]  >= threshold[j]]
                    diff_split[j+1] = diff[dset][diff[dset] >= threshold[j]]

            meas[dset] = meas_split
            hrrr[dset] = hrrr_split
            nam[dset]  = nam_split
            diff[dset] = diff_split
        
        A = {} # create data matrices
        for dset in ['train', 'test']:
            A[dset] = [numpy.array([], dtype=numpy.float)]*n_parts
            for j in xrange(n_parts):
                A[dset][j] = numpy.vstack([hrrr[dset][j], nam[dset][j]]).T  
                if center:
                    A[dset][j] = numpy.hstack([A[dset][j], numpy.ones((A[dset][j].shape[0],1))])

            print dset
            psizes = map(len, diff[dset])
            psum = float(sum(psizes))
            print 'size of partitions: ', psizes
            print 'partitions perc: ', map(lambda x: len(x)/psum, diff[dset])
        
        
        
        w = [numpy.array([], dtype = numpy.float)]*n_parts
        error_vec = {'mix' : numpy.array([], dtype=numpy.float),
                    'hrrr' : numpy.array([], dtype=numpy.float),
                    'nam' : numpy.array([], dtype=numpy.float)}
        for j in xrange(n_parts):
            # solve for least squares weights with training data
            C = numpy.dot(A['train'][j].T, A['train'][j]) + l2reg * numpy.eye(A['train'][j].shape[1])
            b = numpy.dot(A['train'][j].T, meas['train'][j])
            w[j] = numpy.linalg.solve(C, b)

            # measure test errore
            error_vec['mix'] = numpy.append(error_vec['mix'], numpy.dot(A['test'][j],w[j]) - meas['test'][j])
            error_vec['hrrr'] = numpy.append(error_vec['hrrr'], hrrr['test'][j] - meas['test'][j])
            error_vec['nam'] = numpy.append(error_vec['nam'], nam['test'][j] - meas['test'][j])

            error_part['mix'][j][i] = numpy.sqrt(numpy.mean((meas['test'][j] - numpy.dot(A['test'][j],w[j]))**2))
            error_part['hrrr'][j][i] = numpy.sqrt(numpy.mean((meas['test'][j] - hrrr['test'][j])**2))
            error_part['nam'][j][i] = numpy.sqrt(numpy.mean((meas['test'][j] - nam['test'][j])**2))

        error_total['mix'][i] = numpy.sqrt(numpy.mean((error_vec['mix'])**2))
        error_total['hrrr'][i] = numpy.sqrt(numpy.mean((error_vec['hrrr'])**2))
        error_total['nam'][i] = numpy.sqrt(numpy.mean((error_vec['nam'])**2))
    
    for i in xrange(n_parts):
        print 'partition: ', i
        print 'average mixture error: ', numpy.mean(error_part['mix'][i])
        print 'average hrrr error: ', numpy.mean(error_part['hrrr'][i])
        print 'average nam error: ', numpy.mean(error_part['nam'][i])
 
    print 'total average mixture error: ', numpy.mean(error_total['mix'])
    print 'total average hrrr error: ', numpy.mean(error_total['hrrr'])
    print 'total average nam error: ', numpy.mean(error_total['nam'])


def difference_dist(path_root = 'prediction'):

    paths = glob.glob(path_root + '/*.csv')
    
    diffs = numpy.array([], dtype = numpy.float)
    for i,p in enumerate(paths):
        
        df = pandas.read_csv(p)
        
        hrrr = df['irrMoOp1_Total']    
        nam = df['irrMoOp2_Total']
        diffs = numpy.append(diffs, numpy.abs(hrrr-nam))
    
    n, bins, _ = pyplot.hist(diffs, 100)
    n = numpy.cumsum(n / float(numpy.sum(n)))
    pyplot.savefig('plots/nam-hrr-diffs.pdf')

    print n, bins
    
    
def get_rmse(A, w, b):
    return numpy.sqrt(numpy.mean((numpy.dot(A,w) - b)**2))
    
if __name__ == '__main__':
    #plac.call(split_mixture)
    plac.call(main)
    #difference_dist()

