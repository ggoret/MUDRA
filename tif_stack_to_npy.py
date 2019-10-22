#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

__author__ = "Gael Goret"
__copyright__ = "Copyright 2015, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"


import numpy as np
import fabio
import sys
import time

 
def progress(count, total, start_time):
    try: assert count!=0
    except: count = 0.000001
    bar_len = 50
    filled_len = int(round(bar_len * count / total))
    percents = 100.0 * count / total
    timetaken = time.time() - start_time
    estimated_remaining_time = (timetaken/count) * (total - count) 
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s | %.1fs, %.1fs \r' % (bar, percents, '%', timetaken , estimated_remaining_time))
    if percents >= 100:
        print('[%s] %s%s | %.1fs, %.1fs \r' % (bar, percents, '%', timetaken , estimated_remaining_time))     

def main():
    output_fname = sys.argv[1]
    flist = sys.argv[2:]
    stest = fabio.open(flist[0])
    dim1, dim2 = stest.data.shape[0], stest.data.shape[1]
    dim3 = len(flist)
    dims = (dim1, dim2, dim3)
    print('initializing data cube with shape = ',dims)
    cube = np.zeros(dims, stest.data.dtype)
    print('Stacking image files')
    t0 = time.time()
    for i, fname in enumerate(flist):
        progress(i, dim3-1, t0)
        img = fabio.open(fname)
        assert (img.data.shape[0] == dim1 and img.data.shape[1] == dim2 and img.data.dtype == stest.data.dtype), 'Dimensions or data type incompatible for file %s'%fname
        cube[:,:,i] = img.data
    np.save(output_fname,cube)
    
            
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python tif_stack_to_npy.py output_fname images-file(s)')
    else:
        main()
    sys.exit()
