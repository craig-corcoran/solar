import re
import plac
import numpy
import pandas
import pygrib

def main(path = 'forecast/rap/20130501/1312110000200',
        positions = [(40.05,-88.37),
                     (40.05,-88.37),
                     (40.05,-88.37)]):
    
    grib_file = pygrib.open(path)
    
    # get lat/lon as it is the same for all variables
    lat, lon = grib_file.next().latlons()
    flat_lat = lat.flatten()
    flat_lon = lon.flatten()

    df = pandas.DataFrame(numpy.vstack([flat_lat, flat_lon]).T,
                           columns = ['latitude', 'longitude'])
    
    #pos = (40.05,-88.37)
    #positions = zip(flat_lat, flat_lon)
    #for p in positions:
        #dist = numpy.linalg.norm((p[0] - pos[0], p[1] - pos[1]))
        #if dist < 0.1:
            #print p

    #print grib_file.next().keys()
    
    grib_file.seek(0)
    for g in grib_file:
    
        name = g['name']
        ma_level = re.search('(levels|level) (\d+[\. -]?\d*)', str(g))
        ma_fcst = re.search('(fcst time) (.*):', str(g))
        for ma in [ma_level, ma_fcst]:

            if ma is not None:
                strin = ma.groups()[0] + ':' + ma.groups()[1]
            if strin is None:
                print ma.group()
                print ma.groups()
                assert False
            name = name + strin
        
        print name
        #print str(g)

        if df.get(name) is None:
            df[name] = g['values'].flatten()
        else:
            if not ('unknown' in name):
                print 'duplicate: ', name
                print str(g)
                print g['values'].flatten()
                print df[name]
                #assert False

        #print g
        #print 'short name: ', g['shortName']
        #print 'name: ', g['name']
        #print 'paramID: ', g['paramId']
        #print 'param number: ', g['parameterNumber']
        #print 'param name: ', g['parameterName']

        #print 'level: ', g['level']
        #print 'count: ', g['count']
        #print 'identifier: ', g['identifier']
        #print 'type of level: ', g['typeOfLevel']
        #print 'bottom level: ', g['bottomLevel']
        #print 'top level: ', g['topLevel']
        #print 'section number: ', g['sectionNumber']
        #print 'number of section: ', g['numberOfSection']
        ##print 'day: ', g['day']
        #print 'hour: ', g['hour']
        #print 'minute: ', g['minute']
        #print 'dataDate: ', g['dataDate']
        #print 'dataTime: ', g['dataTime']
        #print 'forecast time', g['forecastTime']
        #print 'resolution and component flags1: ', g['resolutionAndComponentFlags1']
        #print 'resolution and component flags2: ', g['resolutionAndComponentFlags2']

    print df

def test_grib(path = 'rap/20130501/1312110000100'):
    ''' tests that all variables in a given RAP grib file have the same 
    lat/lon grids '''
    grib_file = pygrib.open(path)
    f_lat, f_lon = grib_file.next().latlons()
    for g in grib_file:
        lat, lon = g.latlons()
        assert (lat == f_lat).all() & (lon == f_lon).all()

if __name__ == '__main__':
    #test_grib()
    plac.call(main)
