import multiprocessing as mp
import numpy
from scipy import interpolate

def interpolate_grid(it, n_points, grid_dim):
    
    print 'interpolating: ', it
    
    numpy.random.seed(it)

    points = 2*numpy.random.random((n_points,2))-0.5
    values = numpy.random.random(n_points)
    x,y = numpy.mgrid[0:1:grid_dim*1j,0:1:grid_dim*1j]
    grid_pnts = numpy.vstack([x.flatten(), y.flatten()]).T
    interp_vals = interpolate.griddata(points, values, grid_pnts, 'cubic')
    print 'done interpolating: ', it
    return interp_vals
        
def interpolation_test(n_runs = 10, n_points = 1000, grid_dim = 10, parallel = True):
    
    class Sum(object):
        def __init__(self, grid_dim):
            self.acc = numpy.zeros(grid_dim*grid_dim)

        def accumulate(self, val):
            self.acc += val
    
    acc = Sum(grid_dim)

    if parallel:
        pool = mp.Pool(mp.cpu_count()/2)
        for i in xrange(n_runs):
            pool.apply_async(interpolate_grid, 
                            args = [i, n_points, grid_dim], 
                            callback = acc.accumulate)
        pool.close()
        pool.join()

    else:
        for i in xrange(n_runs):
            acc.acc += interpolate_grid(i, n_points, grid_dim)

    return acc.acc

def main():
    
    ser_result = interpolation_test(parallel = False)
    par_result = interpolation_test(parallel = True)
    
    assert ((par_result-ser_result) < 1e-14).all()
    print 'serial and parallel results the same'

if __name__ == '__main__':
    main()
