import plac
import glob
import pandas
import numpy
from matplotlib import pyplot

class MixtureModel(object):
    
    def __init__(self, A, target, mask, l2reg = 0., center = True):
        
        self.center = center
        if center: # append constant column
            A = numpy.hstack([A, numpy.ones((A.shape[0],1))])

        not_mask = numpy.array(map(lambda x: not x, mask))
        self.target = {'train': target[mask], 'test': target[not_mask]} 
        self.A  = {'train': A[mask,:], 'test': A[not_mask,:]} 
        
        self.l2reg = l2reg
        C = numpy.dot(self.A['train'].T, self.A['train']) + l2reg * numpy.eye(self.n_params)
        b = numpy.dot(self.A['train'].T, self.target['train'])
        self.w = numpy.linalg.solve(C, b)

    @property
    def n_params(self):
        return int(self.A['train'].shape[1])

    def model_errors(self):
        ''' returns dictionaries with the test and training RMSE and standard 
        error for each column in A, which assumes each column is a model trying 
        to predict the target values
        '''
        mn_err = {}; mn_std = {}
        for dset in ['train', 'test']:
            err = (self.A[dset] - numpy.repeat(self.target[dset][:,None], self.n_params, axis = 1))**2
            if self.center:
                err = err[:, :-1]
            mn_err[dset] = numpy.sqrt(numpy.mean(err, axis = 0))
            mn_std[dset] = numpy.sqrt(numpy.std(err, axis = 0)) #/ numpy.sqrt(len(self.target[dset]))
        return mn_err, mn_std
    
    def mixture_errors(self):
        ''' returns dictionaries with the test and training RMSE and standard 
        error for the mixture using the learned weights w.
        '''
        mn_err = {}; mn_std = {}
        for dset in ['train', 'test']:
            err = (numpy.dot(self.A[dset], self.w) - self.target[dset])**2
            mn_err[dset] = numpy.sqrt(numpy.mean(err, axis = 0))
            #std_err[dset] = numpy.std(err, axis = 0) / numpy.sqrt(len(self.target[dset]))
            mn_std[dset] = numpy.sqrt(numpy.std(err, axis = 0))
        return mn_err, mn_std

class MixtureData(object):
    
    def __init__(self):

        self.columns = ['model', 'fold', 'l2reg',
            'n_samples-test', 'n_samples-train',
            'error-test', 'error-train',
            'std-test', 'std-train', 
            'site_index']
        self.df = pandas.DataFrame(columns = self.columns)

    def add_row(self, vals):
        row = pandas.DataFrame([dict(zip(self.columns, vals)), ])
        self.df = self.df.append(row, ignore_index=True)  


    def record_errors(self, m, fold, n_samp, site_ind, models):
        er, se = m.mixture_errors()
        er_mod, se_mod = m.model_errors()
        
        vals = ['mix', fold, m.l2reg,
                n_samp[0], n_samp[1], 
                er['test'], er['train'], 
                se['test'], se['train'], site_ind]
        assert len(vals) == len(self.columns)
        self.add_row(vals)

        for j, mod in enumerate(models):
            vals = [mod, fold, m.l2reg,
                n_samp[0], n_samp[1], 
                er_mod['test'][j], er_mod['train'][j], 
                se_mod['test'][j], se_mod['train'][j], site_ind]
            assert len(vals) == len(self.columns)
            self.add_row(vals)

def plot_average(d_avg_error, d_avg_stder, title, keys, width = 0.1,
                ind = None, sp_num = None, colors = None):

    sp_num = sp_num if sp_num else 111 
    ind = ind if ind else 1
    colors = colors if colors else ['b','g','r','c','m','y']

    
    ax = pyplot.subplot(sp_num)
    
    n_mods = len(d_avg_error.keys())
    offset = -width * (n_mods / 2)
    for i, mod in enumerate(keys):
        ax.bar(ind + offset + (width*i), 
            d_avg_error[mod], 
            width = width / 2.,  
            align = 'center', 
            yerr = d_avg_stder[mod], 
            ecolor = 'k', 
            color = colors[i % len(colors)])
    ax.autoscale(tight=False)
    pyplot.xticks([])
    pyplot.box(on = False)
    pyplot.title(title, fontsize = 20, va='top')


def main(models = ['max_diurnal', 'mean_diurnal', 'rap', 'hrrr', 'nam'],
        col_names = ['Max Diurnal Forecast','Mean Diurnal Forecast',
                    'RAP Forecast', 'HRRR Forecast','NAM Forecast','Actual']):
    
    # arm crossval experiment
    print 'performing crossval experiment on ARM data'
    d_avg_error, d_avg_stder, \
    d_site_error, d_site_stder = mixture_experiment(
                regex = 'data/prediction/arm-5mixture.csv',
                n_folds = 10,
                models = models, 
                col_names = col_names)

    for i,dset in enumerate(['train','test']):
        title = 'Crossvalidation - %s' % dset
        keys = numpy.array(d_avg_error.keys())
        keys = keys[numpy.array([dset in k for k in keys])]
        keys = sorted(keys)
        subplot_num = 321 + i
        plot_average(d_avg_error, d_avg_stder, title, keys, sp_num = subplot_num)
        
        legend_names = map(lambda x: x.replace('_',' '), keys)
        pyplot.legend(legend_names, loc=2)
    

    # arm forecast experiment
    print 'performing forecast experiment on ARM data'
    hold_outs = [(5,16), (6,1)]
    for j, h in enumerate(hold_outs):
        d_avg_error, d_avg_stder, \
        d_site_error, d_site_stder = mixture_experiment(
                    regex = 'data/prediction/arm-5mixture.csv',
                    n_folds = None,
                    hold_out = h,
                    models = models, 
                    col_names = col_names) 
        
        for i,dset in enumerate(['train','test']):
            title = 'Forecast %s onward - %s' % (str(h).replace('(','').replace(')','').replace(',','/').replace(' ',''), dset)
            keys = numpy.array(d_avg_error.keys())
            keys = keys[numpy.array([dset in k for k in keys])]
            keys = sorted(keys)
            subplot_num = 323 + j*2 + i
            plot_average(d_avg_error, d_avg_stder, title, keys, sp_num = subplot_num)

            #legend_names = map(lambda x: x.replace('_',' '), keys)
            #pyplot.legend(legend_names, loc=2)
    
    suptitle = 'ARM 5 Model Mixture Error'
    pyplot.suptitle(suptitle, fontsize = 20)
    pyplot.savefig('plots/%s.pdf' % suptitle.replace(' ', '_').lower())

    # XXX plot weights


def mixture_experiment(
    regex = 'data/prediction/arm-5mixture.csv', # 'data/prediction/20130606/*.csv',
    to_predict = None, #'Total', #'Diff',# 'DirNorm', #
    l2reg = 0., 
    hold_out = None, # (6,1), # (5,16), #
    n_folds = 10, # None, # 
    center = True,
    models = ['max_diurnal', 'mean_diurnal', 'rap', 'hrrr', 'nam'], #['hrrr','nam'],
    col_names = ['Max Diurnal Forecast','Mean Diurnal Forecast','RAP Forecast',
                'HRRR Forecast','NAM Forecast','Actual']): #['irrMoOp1_', 'irrMoOp2_','irrMe']):
    
    # last column name is target/measured
    assert len(col_names) == (len(models) + 1)
    
    # append to_predict if present
    col_names = map(lambda x: (x + to_predict) if to_predict else x, col_names)

    paths = glob.glob(regex)
    paths = sorted(paths)
    
    err_data = MixtureData()

    for i,p in enumerate(paths):
        
        print 'file: ', p

        df = pandas.read_csv(p)

        d_columns = {'meas': df[col_names[-1]]}
        for i,mod in enumerate(models):
            d_columns[mod] = df[col_names[i]]
        
        # model data matrix
        A = numpy.vstack([d_columns[m] for m in models]).T
            
        n_samp = len(d_columns['meas']) 
        if (hold_out is None) & (n_folds > 0):
            for f in xrange(n_folds):

                ind_a = numpy.round(n_samp*f / float(n_folds)).astype(int)
                ind_b = numpy.round(n_samp*(f+1) / float(n_folds)).astype(int)
                mask = numpy.array([True]*n_samp)
                mask[ind_a:ind_b] = False
                
                n_samp_tup = (n_samp-sum(mask), sum(mask))
                
                # build model, measure errors
                m = MixtureModel(A, d_columns['meas'], mask, l2reg, center)
                err_data.record_errors(m, f, n_samp_tup, i, models)
                
                
        elif (n_folds is None) & (len(hold_out) == 2):
            print 'holding out: ', hold_out
            # find the first data point after hold out day
            mon_ind = numpy.where(df['month']==hold_out[0])[0][0]
            day_ind = numpy.where(df['day']==hold_out[1])[0]
            ind = day_ind[numpy.where(day_ind > mon_ind)[0][0]]
            mask = numpy.array([True]*n_samp)
            mask[ind:] = False
            n_samp_tup = (n_samp-sum(mask), sum(mask))

            m = MixtureModel(A, d_columns['meas'], mask, l2reg, center)
            err_data.record_errors(m, 1, n_samp_tup, i, models)
            
        else: 
            print 'n_folds or hold_out date must be set and the other None'
            assert False
            
    err_data.df.to_csv('%s-error.csv' % regex.split('.')[0])

    nf = n_folds if n_folds else 1.

    # aggregate data for each model, site pair
    gb = err_data.df.groupby(['model', 'site_index'])
    agg = gb.aggregate(numpy.mean)

    d_site_error = {}; d_site_stder = {} 
    d_avg_error = {}; d_avg_stder = {}
    
    for mod in models + ['mix']:
        for ds in ['-train', '-test']:
            
            frame = agg.ix[mod]
            mn_samps = frame['n_samples' + ds]
            mn_err = frame['error' + ds]
            mn_std = frame['std' + ds]
            mn_stder = mn_std / (numpy.sqrt(mn_samps * nf))

            d_site_error[mod + ds] = mn_err
            d_site_stder[mod + ds] = mn_stder

            d_avg_error[mod + ds] = (mn_err * mn_samps).sum() / mn_samps.sum()
            d_avg_stder[mod + ds] = (mn_stder * mn_samps).sum() / mn_samps.sum()

        print 'mean %s test error: ' % mod,  d_avg_error[mod + '-test']
        print 'mean %s test standard error: ' % mod,  d_avg_stder[mod + '-test']

    return d_avg_error, d_avg_stder, d_site_error, d_site_stder


    # plotting
    
        

    # get per-day error?

    # plotting

    #if plot:

        #if to_predict is 'Diff':
            #to_predict = 'Diffuse'
        #elif to_predict is 'DirNorm':
            #to_predict = 'Direct Normal'

        #ind = numpy.arange(n_sites)
        
        ## plot errors by site
        #pyplot.clf()
        #ax = pyplot.subplot(111)
        #ax.bar(ind-0.2, d_site_error['mix'], width=0.2, color='g', align='center', yerr=d_site_stder['mix'], ecolor='k')
        #ax.bar(ind, d_site_error['hrrr'], width=0.2, color='b', align='center', yerr=d_site_stder['hrrr'], ecolor='k')
        #ax.bar(ind+0.2, d_site_error['nam'], width=0.2, color='r', align='center', yerr=d_site_stder['nam'], ecolor='k')
        #pyplot.legend(['LR Mixture', 'HRRR', 'NAM'], fontsize = 20, loc=2)
        #site_codes = ['bon','tbl','dra','fpk','gwn','psu','sxf','abq','bis','hxn','msn','sea','slc','ste']
        #pyplot.xticks(ind, site_codes, size = 25, fontweight='bold')
        #pyplot.ylabel('RMSE $(W/m^2)$', size = 20, fontweight='bold')
        #pyplot.suptitle('%s Irradiance %s Error by Site' % 
                #(to_predict, 'Forecast' if n_folds is None else 'Crossval'), 
                #fontsize = 30, fontweight='bold')
        #ax.autoscale(tight=False)
        #pyplot.savefig('plots/site_error_%s-%s.pdf' % (to_predict.replace(' ',''),'Forecast' if n_folds is None else 'Crossval'))

        ## plot learned weights by site
        #pyplot.clf()
        #ax = pyplot.subplot(211)
        #ax.bar(ind-0.1, weights['hrrr'], width=0.2, color='b', align='center')
        #ax.bar(ind+0.1, weights['nam'], width=0.2, color='r', align='center')
        #pyplot.legend(['HRRR', 'NAM'], fontsize = 20, loc=2)
        #pyplot.xticks(ind, ['']*14, size = 25)
        #pyplot.ylabel('Regression Weight', size = 20, fontweight='bold')
        #pyplot.suptitle('%s Linear Regression Weights by Site - %s' % 
                    #('Forecast' if n_folds is None else 'Crossval', to_predict),
                    #fontsize = 30, fontweight='bold')
        #ax.autoscale(tight=False)

        #ax = pyplot.subplot(212)
        #ax.bar(ind, weights['const'], width=0.2, color='k', align='center')
        #pyplot.legend(['Const'], fontsize = 20,  loc=0)
        #pyplot.xticks(ind, site_codes, size = 25, fontweight='bold')
        #pyplot.ylabel('Irradiance $(W/m^2)$', size = 20, fontweight='bold')
        #ax.autoscale(tight=False)
        #pyplot.savefig('plots/site_weights_%s-%s.pdf' % \
            #(to_predict.replace(' ',''),'Forecast' if n_folds is None else 'Crossval'))

    
if __name__ == '__main__':
    plac.call(main)


