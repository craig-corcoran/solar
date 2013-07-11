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
        reg_matrix = self.l2reg * numpy.eye(self.n_params)
        reg_matrix[-1,-1] = 0.
        C = numpy.dot(self.A['train'].T, self.A['train']) + reg_matrix
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
        
        # add mixture error value
        vals = ['mix', fold, m.l2reg,
                n_samp[0], n_samp[1], 
                er['test'], er['train'], 
                se['test'], se['train'], site_ind]
        assert len(vals) == len(self.columns)
        self.add_row(vals)
        
        # add models error values
        for j, mod in enumerate(models):
            vals = [mod, fold, m.l2reg,
                n_samp[0], n_samp[1], 
                er_mod['test'][j], er_mod['train'][j], 
                se_mod['test'][j], se_mod['train'][j], site_ind]
            assert len(vals) == len(self.columns)
            self.add_row(vals)

def mixture_experiment(
    regex = 'data/prediction/20130606/*.csv', # 'data/prediction/arm-6mixture.csv', # 
    to_predict = None, #'Total', #'Diff',# 'DirNorm', #
    l2reg = 0., 
    hold_out = None, # (6,1), # (5,16), #
    n_folds = 10, # None, # 
    center = True,
    models = ['eulerian', 'max_diurnal', 'mean_diurnal', 'rap', 'hrrr', 'nam'],
    col_names = ['Eulerian', 'Max Diurnal Forecast','Mean Diurnal Forecast','RAP Forecast',
                'HRRR Forecast','NAM Forecast','Actual']): 
    
    # last column name is target/measured
    assert len(col_names) == (len(models) + 1)
    
    # append to_predict if present
    col_names = map(lambda x: (x + to_predict) if to_predict else x, col_names)

    paths = glob.glob(regex)
    paths = sorted(paths)
    
    err_data = MixtureData()
    avg_weight = None

    for i,p in enumerate(paths):
        
        print 'file: ', p
        print 'reg: ', l2reg

        df = pandas.read_csv(p)
        #print 'pre processing frame: ', df

        for col in col_names:
            dat = df[col]
            if sum(dat.notnull()) < len(dat):
                df = df.ix[dat.notnull()]

        #print 'post processing', df

        d_columns = {'meas': df[col_names[-1]]}
        for j,mod in enumerate(models):
            d_columns[mod] = df[col_names[j]]
        
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

                m = MixtureModel(A, d_columns['meas'], mask, l2reg, center)
                err_data.record_errors(m, f, n_samp_tup, i, models)
                avg_weight = m.w if avg_weight is None else avg_weight + m.w

                #print m.w
                
                
        elif (n_folds is None) & (len(hold_out) == 2):
            print 'holding out: ', hold_out
            # find the first data point after hold out day
            mon_ind = numpy.where(df['month']==hold_out[0])[0][0]
            day_ind = numpy.where(df['day']==hold_out[1])[0]
            
            print 'days index: ', numpy.where(day_ind > mon_ind)
            ind = day_ind[numpy.where(day_ind > mon_ind)[0][0]]
            mask = numpy.array([True]*n_samp)
            mask[ind:] = False
            n_samp_tup = (n_samp-sum(mask), sum(mask))

            m = MixtureModel(A, d_columns['meas'], mask, l2reg, center)
            err_data.record_errors(m, 1, n_samp_tup, i, models)
            avg_weight = m.w if avg_weight is None else avg_weight + m.w

            print m.w
            
        else: 
            print 'n_folds or hold_out date must be set and the other None'
            assert False
            
    nf = n_folds if n_folds else 1.
    avg_weight = avg_weight / (len(paths) * nf)
    
    # write error data to file
    #err_data.df.to_csv('%s-error.csv' % regex.split('.')[0])

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

        print '%s test error: ' % mod,  d_avg_error[mod + '-test']
        print '%s test standard error: ' % mod,  d_avg_stder[mod + '-test']

    return d_avg_error, d_avg_stder, d_site_error, d_site_stder, avg_weight

def plot_bars(d_bar, d_err, title, keys, width = 0.1,
                ind = None, sp_num = None, colors = None, share_ax = None):
    sp_num = sp_num if sp_num else 111 
    ind = ind if ind else 1
    colors = colors if colors else ['b','g','r','c','m','y']
    
    if share_ax is None:
        ax = pyplot.subplot(sp_num)
    else:
        ax = pyplot.subplot(sp_num, sharey = share_ax, sharex = share_ax)
    
    n_mods = len(keys)
    offset = -width * (n_mods / 2)
    for i, mod in enumerate(keys):
        ax.bar(ind + offset + (width*i), 
            d_bar[mod], 
            width = width / 2.,  
            align = 'center', 
            yerr = d_err[mod] if d_err else None, 
            ecolor = 'k', 
            color = colors[i % len(colors)])
    #ax.autoscale(tight=False)
    pyplot.xticks([])
    pyplot.box(on = False)
    pyplot.title(title, fontsize = 20, va='baseline')
    pyplot.legend(keys)

    return ax

def single_model(path_regex = 'data/prediction/20130606/*.csv', #'data/prediction/arm-6mixture.csv',
        models = ['hrrr', 'nam'],
        col_names = ['irrMoOp1_','irrMoOp2_','irrMe'],
        to_predict = 'Total', # 'Diff' # 'DirNorm' #
        l2reg = 0., #100000.,
        suptitle =  'Single Models with Scaling and Bias'):
        
        col_names = map(lambda x: x + to_predict, col_names)
            
        pyplot.clf()
        ax = None
        for i,mod in enumerate(models):
            
            keys = [mod+'-test', 'mix-test']
            
            print 'performing crossval experiment %s' % mod
            d_avg_error, d_avg_stder, \
            d_site_error, d_site_stder, \
            avg_weight_cv = mixture_experiment(
                regex = path_regex,
                n_folds = 10,
                models = [mod], 
                col_names = [col_names[i]] + [col_names[-1]],
                l2reg = l2reg)

            print d_site_error.keys()
            print d_avg_error.keys()

            sp_num = 131+i
            if ax is None:
                ax = plot_bars(d_avg_error, d_avg_stder, mod + ' only', keys, width = 0.2, sp_num = sp_num)
            else:
                plot_bars(d_avg_error, d_avg_stder, mod + ' only', keys, width = 0.2, sp_num = sp_num, share_ax = ax)


        d_avg_error, d_avg_stder, \
        d_site_error, d_site_stder, \
        avg_weight_cv = mixture_experiment(
                regex = path_regex,
                n_folds = 10,
                models = models, 
                col_names = col_names,
                l2reg = l2reg)
        
        keys = ['mix-test']
        plot_bars(d_avg_error, d_avg_stder, ', '.join(models) + ' mixture', 
                    keys, width = 0.2, sp_num = 133, colors = ['g'], share_ax = ax)
        pyplot.savefig('single-model.pdf')
            
        




def main(path_regex = 'data/prediction/20130606/*.csv', #'data/prediction/arm-6mixture.csv',
        models = ['eulerian', 'max_diurnal', 'mean_diurnal', 'rap', 'hrrr', 'nam'],
        col_names = ['Eulerian', 'Max Diurnal Forecast','Mean Diurnal Forecast',
                 'RAP Forecast', 'HRRR Forecast','NAM Forecast','Actual'],
        forecast = False,
        l2reg = 100000.,
        suptitle =  'ARM 6 Model Mixture'):
    
    # arm crossval experiment
    print 'performing crossval experiment on ARM data'
    d_avg_error, d_avg_stder, \
    d_site_error, d_site_stder, \
    avg_weight_cv = mixture_experiment(
                regex = path_regex,
                n_folds = 10,
                models = models, 
                col_names = col_names,
                l2reg = l2reg)
    pyplot.clf()

    for i,dset in enumerate(['train','test']):
        title = 'Crossvalidation - %s' % dset
        keys = numpy.array(d_avg_error.keys())
        keys = keys[numpy.array([dset in k for k in keys])]
        keys = sorted(keys)
        subplot_num = 321 + i if forecast else 121 + i
        plot_bars(d_avg_error, d_avg_stder, title, keys, sp_num = subplot_num)
        
        legend_names = map(lambda x: x.replace('_',' '), keys)
        pyplot.legend(legend_names, loc=2)
    

    # arm forecast experiments
    if forecast:
        print 'performing forecast experiment on ARM data'
        hold_outs = [(5,16), (6,1)]
        avg_weight_fc = [None]*len(hold_outs)
        for j, h in enumerate(hold_outs):
            d_avg_error, d_avg_stder, \
            d_site_error, d_site_stder, \
            avg_weight_fc[j] = mixture_experiment(
                        regex = path_regex,
                        n_folds = None,
                        hold_out = h,
                        models = models, 
                        col_names = col_names,
                        l2reg = l2reg) 

            for i,dset in enumerate(['train','test']):
                title = 'Forecast %s onward - %s' % (str(h).replace('(','').replace(')','').replace(',','/').replace(' ',''), dset)
                keys = numpy.array(d_avg_error.keys())
                keys = keys[numpy.array([dset in k for k in keys])]
                keys = sorted(keys)
                subplot_num = 323 + j*2 + i
                plot_bars(d_avg_error, d_avg_stder, title, keys, sp_num = subplot_num)

                #legend_names = map(lambda x: x.replace('_',' '), keys)
                #pyplot.legend(legend_names, loc=2)
        
    pyplot.suptitle(suptitle + ' Error', fontsize = 20)
    pyplot.savefig('plots/%s_error.reg=%s.pdf' % (suptitle.replace(' ', '_').lower(), str(l2reg)))
    
    # plot crossval weights    
    pyplot.clf()
    d_weights = dict(zip(models, avg_weight_cv[:-1]))
    keys = sorted(models)
    title = 'Average Crossvalidation Weights' 
    sp_num = 131 if forecast else 111
    ax_cv = plot_bars(d_weights, None, title, keys, sp_num = sp_num)

    ymin, ymax = pyplot.ylim()
    
    legend_names = map(lambda x: x.replace('_',' '), keys)
    pyplot.legend(legend_names, fontsize=20, loc = 4)

    
    if forecast:
        # plot forecast weights    
        axes = [None] * len(hold_outs)
        axes.append(ax_cv)
        for j, h in enumerate(hold_outs):
            d_weights = dict(zip(models, avg_weight_fc[j][:-1]))
            title = 'Forecast Weights - %s onward' % str(h).replace('(','').replace(')','').replace(',','/').replace(' ','')
            axes[j] = plot_bars(d_weights, None, title, sorted(models), sp_num = 132 + j)

            ymin_fc, ymax_fc = pyplot.ylim()
            ymin = min(ymin, ymin_fc)
            ymax = max(ymax, ymax_fc)
            
        for a in axes:
            lims = a.axis()
            a.axis([lims[0], lims[1], ymin, ymax])

    pyplot.suptitle(suptitle + ' Weights', fontsize = 20)
    pyplot.savefig('plots/%s_weights.reg=%s.pdf' % (suptitle.replace(' ', '_').lower(), str(l2reg)))

    
if __name__ == '__main__':
    #plac.call(main)
    plac.call(single_model)


