import numpy
import pandas as pd

def read_surfrad_dat(ftp_string, nrows, skiprows, names, delim_whitespace = True):
    table = pd.read_csv( 
                    filepath_or_buffer = ftp_string,
                    delim_whitespace = delim_whitespace,
                    nrows = nrows,
                    skiprows = skiprows,
                    header = None,
                    na_values = [-9999.9],
                    names = names)
                    #skipinitialspace = True) # doesnt seem to work, still returning NaN for first column
    # fix the index from being NaN
    table.index = numpy.linspace(0,len(table.index)-1,len(table.index)).astype(int)
    return table

# for units and notes on abreviations, see
# ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/Boulder_CO/README
line2_names = ['latitude', 'longitude', 'elevation', 'version']
header_names = ['year', 
                'julian_day', 
                'month', 
                'day', 
                'hour', 
                'min', 
                'decimal_time', 
                'zenith_angle', 
                'dw_solar', 'qc_dw_solar', 
                'uw_solar', 'qc_uw_solar',
                'direct_n', 'qc_direct_n',
                'diffuse', 'qc_diffuse',
                'dw_ir', 'qc_dw_ir',
                'dw_casetemp', 'qc_dw_casetemp',
                'dw_dometemp', 'qc_dw_dometemp',
                'uw_ir', 'qc_uw_ir',
                'uw_casetemp', 'qc_uw_casetemp',
                'uw_dometemp', 'qc_uw_dometemp',
                'uvb', 'qc_uvb',
                'par', 'qc_par',
                'netsolar', 'qc_netsolar',
                'netir', 'qc_netir',
                'totalnet', 'qc_totalnet',
                'temp', 'qc_temp',
                'rh', 'qc_rh',
                'windspd', 'qc_windspd',
                'winddir', 'qc_winddir',
                'pressure', 'qc_pressure']

# read all but first two lines
# TODO combine ftp_string calls into one file open - urllib2?
#ftp_string = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/Bondville_IL/1995/bon95001.dat'
ftp_string = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/Boulder_CO/2011/tbl11001.dat'

#ftp_string = urllib2.urlopen('ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/Boulder_CO/2011/tbl11001.dat')
table = read_surfrad_dat(ftp_string, None, 2, header_names)

# read the first two lines
line1 = read_surfrad_dat(ftp_string, 1, None, None, False).iloc[0]
line2 = read_surfrad_dat(ftp_string, 1, 1, None).iloc[0]

location = line1[0].lstrip()
lat, lon, elevation = line2[1:4]
version = line2[len(line2)-1]

# add them to the table 
table['location'] = [location] * len(table.index)
table['latitude'] = [lat] * len(table.index)
table['longitude'] = [lon] * len(table.index)
table['elevation'] = [elevation] * len(table.index)
table['version'] = [version] * len(table.index)

print table
print location
print lat
print lon
print elevation
print version


                                    
