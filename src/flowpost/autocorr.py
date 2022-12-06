import numpy as np
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    
    return result[result.size//2:] / (np.var(x)  * len(x))

# the following three are from
# https://stackoverflow.com/questions/16044491/statistical-scaling-of-autocorrelation-using-numpy-fft
import numpy as np
fft = np.fft

def autocorrelation(x):
    """
    Compute autocorrelation using FFT
    The idea comes from
    http://dsp.stackexchange.com/a/1923/4363 (Hilmar)
    """
    x = np.asarray(x)
    N = len(x)
    x = x-x.mean()
    s = fft.fft(x, N*2-1)
    result = np.real(fft.ifft(s * np.conjugate(s), N*2-1))
    result = result[:N]
    result /= result[0]
    return result

def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result

def autocorrelate(x):
    fftx = fft.fft(x)
    fftx_mean = np.mean(fftx)
    fftx_std = np.std(fftx)

    ffty = np.conjugate(fftx)
    ffty_mean = np.mean(ffty)
    ffty_std = np.std(ffty)

    result = fft.ifft((fftx - fftx_mean) * (ffty - ffty_mean))
    result = fft.fftshift(result)
    return [i / (fftx_std * ffty_std) for i in result.real]

# from https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

'''
returns index of first crossing and the integral length scale
'''
def tscale(x, dt=None, threshold=0.2, verbose=False):
    ###################################
    # x must be an autocorrelation function
    # time scales
    ts_ind = np.argmax(x<=threshold)
    if dt is not None:
        ts_int = np.trapz(x[0:ts_ind], dx=dt)
        if verbose:
            print('lag of first threshold crossing for ' + str(threshold) + ' (in samples): ' + str(ts_ind) + ', corresponds to timescale ' + str(ts_int))
        return ts_ind, ts_int
    else:
        if verbose:
            print('lag of first threshold crossing for ' + str(threshold) + ' (in samples): ' + str(ts_ind))
        return ts_ind
#    print 'integral timescale: ' + str(ts_int)

def t_int(ACF):
    '''
    integrated autocorrelation time, according to Sokal (p.7) or Janke (p. 12)
    '''
    mintime = 3
    t = 1
    increment = 1
    ts_int = 0.5
    while (t < len(ACF)):
        # compute normalized fluctuation correlation function at time t
        ts_int = ts_int + ACF[t]
        if (ACF[t] <= 0.2) and (t > mintime):
            break
        t += increment

    return ts_int
