import multiprocessing as mp
import time

def f(x):
    return x*x

def timeit(f,args):
    
    t = time.time()
    val = f(*args)
    t = time.time()-t

    return val,t

if __name__ == '__main__':
    pool = mp.Pool(processes=4)              # start 4 worker processes

    #result = pool.apply_async(f, range(10))    # evaluate "f(10)" asynchronously
    #result, t = timeit(pool.apply_async,[f,10])
    
    r, t = timeit(pool.map, [f, range(10)])
    print t
    print r

    r, t1 = timeit(pool.map_async, [f, range(10)])
    print t1
    val, t2 = timeit(r.get, [])
    print t1+t2
    print val
