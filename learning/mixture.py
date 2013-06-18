import plac
import glob
import pandas
import numpy
from matplotlib import pyplot

def main(
    path_root = 'prediction/20130606',
    to_predict = 'Total', #'Diff',# 'DirNorm', #
    l2reg = 0., 
    hold_out = (5,16),
    center = True,
    abs_diff = False,
    plot = True):

    paths = glob.glob(path_root + '/*.csv')

    n_sites = len(paths)
    error = {'mix' : [0.]*n_sites, 
            'hrrr': [0.]*n_sites, 
            'nam': [0.]*n_sites}
    std_err = {'mix' : [0.]*n_sites, 
               'hrrr': [0.]*n_sites, 
               'nam': [0.]*n_sites}
    weights = {'const' : [0.]*n_sites, 
               'hrrr': [0.]*n_sites, 
               'nam': [0.]*n_sites}


    for i,p in enumerate(paths):
        
        df = pandas.read_csv(p)
        
        meas = df['irrMe%s' % to_predict]    
        hrrr = df['irrMoOp1_%s' % to_predict]    
        nam = df['irrMoOp2_%s' % to_predict]
    
        # find the first data point after hold out day
        may_ind = numpy.where(df['month']==hold_out[0])[0][0]
        day_ind = numpy.where(df['day']==hold_out[1])[0]
        ind = day_ind[numpy.where(day_ind > may_ind)[0][0]]
        
        A = numpy.vstack([hrrr, nam]).T
        if abs_diff:
            A = numpy.hstack([A, numpy.abs(nam - hrrr)[:,None]])
        if center:
            A = numpy.hstack([A, numpy.ones((A.shape[0],1))])

        meas = {'train': meas[:ind], 'test': meas[ind:]} 
        hrrr = {'train': hrrr[:ind], 'test': hrrr[ind:]} # need hrrr/nam training?
        nam = {'train': nam[:ind], 'test': nam[ind:]} 
        A  = {'train': A[:ind], 'test': A[ind:]} 
    
        C = numpy.dot(A['train'].T, A['train']) + l2reg * numpy.eye(A['train'].shape[1])
        b = numpy.dot(A['train'].T, meas['train'])
        w = numpy.linalg.solve(C, b)

        weights['hrrr'][i] = w[0]
        weights['nam'][i] = w[1]
        weights['const'][i] = w[-1]
        
        sq_err = {'mix': (meas['test'] - numpy.dot(A['test'],w))**2,
                  'hrrr': (meas['test'] - hrrr['test'])**2,
                  'nam': (meas['test'] - nam['test'])**2}

        for mod in ['mix', 'hrrr', 'nam']:
            error[mod][i] = numpy.sqrt(numpy.mean(sq_err[mod]))
            std_err[mod][i] = numpy.sqrt(numpy.std(sq_err[mod])) / numpy.sqrt(len(sq_err[mod]))

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
        pyplot.suptitle('%s Irradiance Forecast Error by Site' % to_predict, fontsize = 30, fontweight='bold')
        ax.autoscale(tight=False)
        pyplot.savefig('plots/site_error_%s.pdf' % to_predict.replace(' ',''))

        # plot learned weights by site
        pyplot.clf()
        ax = pyplot.subplot(211)
        ax.bar(ind-0.1, weights['hrrr'], width=0.2, color='b', align='center')
        ax.bar(ind+0.1, weights['nam'], width=0.2, color='r', align='center')
        pyplot.legend(['HRRR', 'NAM'], fontsize = 20, loc=2)
        pyplot.xticks(ind, ['']*14, size = 25)
        pyplot.ylabel('Regression Weight', size = 20, fontweight='bold')
        pyplot.suptitle('Linear Regression Weights by Site - %s' % to_predict, fontsize = 30, fontweight='bold')
        ax.autoscale(tight=False)

        ax = pyplot.subplot(212)
        #ax.bar(ind-0.2, weights['hrrr'], width=0.2, color='b', align='center')
        #ax.bar(ind, weights['nam'], width=0.2, color='r', align='center')
        ax.bar(ind, weights['const'], width=0.2, color='k', align='center')
        pyplot.legend(['Const'], fontsize = 20,  loc=0)
        pyplot.xticks(ind, site_codes, size = 25, fontweight='bold')
        pyplot.ylabel('Irradiance $(W/m^2)$', size = 20, fontweight='bold')
        #pyplot.title('Constant Bias by Site', fontsize = 30)
        ax.autoscale(tight=False)
        pyplot.savefig('plots/site_weights_%s.pdf' % to_predict.replace(' ',''))



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

