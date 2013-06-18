import csv
import plac
import numpy
import pandas

def load_rap(path = 'forecast/rap/RAP.csv',
            max_rows = 50):
    ''' load rap data from csv'''
    with open(path) as rap_file:
        reader = csv.reader(rap_file)

        # init as missing vals
        date, time, model = [numpy.nan]*3 
        lat, lon, dt = [numpy.nan]*3
        ozone, albedo, mask, vegetation, aerosol = [numpy.nan]*5
        params_3d = numpy.zeros((max_rows,7))
        params_3d[:] = numpy.nan
        z_cnt = 0

        df = pandas.DataFrame(index = numpy.arange(max_rows))

        for line in reader:
            if line[0] is '0':
                date, time = line[1].split(':')
                model = line[2]
                #print date, time, model
            elif line[0] is '1':
                lat, lon, dt = line[1:]
                z_cnt = 0
                #print lat, lon, dt 
            elif line[0] is '2': # 2d data
                ozone, albedo, mask, vegetation, aerosol = line[1:]
                #print ozone, albedo, mask, vegetation, aerosol 
            elif line[0] is '3': # 3d data
                # z, cld_frac, wtr_mix, ice_mix, wtr_vap_dens, temp, press
                params_3d[z_cnt,:] = numpy.array(map(float, line[1:]))
                z_cnt += 1
                
            else: assert False

    assert z_cnt == max_rows

    # check that we filled all initialized values
    assert not numpy.isnan(params_3d).any()
    
    names_2d = ['date', 'time', 'model', 'lat', 'lon', 'dt', 
                'ozone', 'albedo', 'mask', 'vegetation', 'aerosol']

    #print 'index length: ', len(df.index)
    #print 'z count: ', z_cnt

    for param in names_2d:
        df[param] = numpy.repeat(eval(param), z_cnt)
    
    names_3d = ['z', 'cld_frac', 'wtr_mix', 'ice_mix', 
                'wtr_vap_dens', 'temp', 'press']
    for i,name in enumerate(names_3d):
        df[name] = params_3d[:,i]
            
    return df

def main():
    load_rap()


if __name__ == '__main__':
    plac.call(main)
