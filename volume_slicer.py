from __future__ import division, absolute_import, print_function, unicode_literals # Ready for the future !

__author__ = "Gael Goret"
__copyright__ = "Copyright 2016, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class slicer(object):
    def __init__(self, data, axis, init_slice):
        self.axis = axis
        self.axis_label = {0:'X', 1:'Y', 2:'Z'}[axis]
        
        self.data = np.swapaxes(data, self.axis, 0)
        self.slice_index = init_slice
        self.slice = self.data[self.slice_index]
        self.max_index = self.data.shape[0]
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.slice.T, interpolation='none', origin='lower')
        self.along_axis = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slab = Slider(self.along_axis, '%s_Slab'%self.axis_label, 0, self.max_index , valinit=self.slice_index,  valfmt='%i')
        self.slab.on_changed(self.update_figure)
        self.fig.canvas.mpl_connect('key_press_event',self.update_slice_index)
        plt.show()
        
    def draw(self):
        im = your_function(self.values)
        pylab.show()
        self.ax.imshow(im)

    def update_slice_index(self, event):
        if event.key=='+':
            self.slice_index += 1
        elif event.key == '-':
            self.slice_index -= 1
            if self.slice_index < 0:
                self.slice_index = self.max_index
        self.slab.set_val(self.slice_index)
        
    def update_figure(self, event = None):
        self.slice_index = int(self.slab.val%self.max_index)
        self.slice = self.data[self.slice_index]
        self.ax.imshow(self.slice.T, interpolation='none', origin='lower')
        self.fig.canvas.draw()

        
class main(object):
    def __init__(self, input_fname, axis, init_slice):
        self.data = np.load(input_fname, mmap_mode='r')
        assert self.data.ndim == 3, "data dimensionality should be 3"
        assert axis in ['X','Y','Z'], "directional axis should be among 'X', 'Y', 'Z'" 
        self.axis = {'X':0, 'Y':1, 'Z':2}[axis]
        assert init_slice < self.data.shape[self.axis], "slice index should be < to %i for chosen axis %s"%(self.data.shape[self.axis], axis)
        self.slicer = slicer(self.data, self.axis, init_slice)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Volume Slicer : simple, stupid volume slicer along orthogonal axes''')
    parser.add_argument('-i', action='store', dest='fname',default='',
                    help='input file (standard NumPy binary file format [*.npy]) containing a 3d array')
    parser.add_argument('-a', action='store', dest='axis',default='X',
                    help="directional axis to slice along, among 'X', 'Y', 'Z' [default=X]")
    parser.add_argument('-s', action='store', dest='init_slice',default=0,
                    help='initial slice index value along the given axis [default=0]')
    proxy = parser.parse_args()
    
    assert proxy.fname != '', parser.parse_args(['-h'])
    
    m = main(proxy.fname, proxy.axis.capitalize(), int(proxy.init_slice))  
   
