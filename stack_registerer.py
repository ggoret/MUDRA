# -*- coding: utf-8 -*-
from __future__ import with_statement, print_function, division # Ready for the future !

__author__ = "Gael Goret"
__copyright__ = "Copyright 2016, CEA"
__version__ = "0.2.5"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"


import numpy as np

from skimage.feature import register_translation
from scipy.ndimage.interpolation import shift

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, Slider

import argparse, os, sys, time


class stack_registerer(object):
    def __init__(self, data, axis, init_slice, mode, fname):
        self.axis = axis
        self.axis_label = {0:'X', 1:'Y', 2:'Z'}[axis]
        self.roi = None
        self.data = np.swapaxes(data, self.axis, 0)
        self.slice_index = init_slice
        self.slice = self.data[self.slice_index]
        self.max_index = self.data.shape[0]
        self.mode = mode
        self.roi = []
        self.shifts = []
        self.errors = []
        self.phasediffs = []
        self.output_fname = fname[:fname.rfind('.')]+'_REG'
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.slice.T, interpolation='none', origin='lower')
        self.along_axis = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slab = Slider(self.along_axis, '%s_Slab'%self.axis_label, 0, self.max_index , valinit=self.slice_index,  valfmt='%i')
        self.slab.on_changed(self.update_figure)
        self.fig.canvas.mpl_connect('key_press_event',self.filter_keys_event)
        self.rs = RectangleSelector(self.ax, self.select_roi, drawtype='box', useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')
                
        print('\n Welcome to PySR : [Py]thon [S]tack [R]egisterer ')
        print('--------------------------------------------------')
        print('\n* Press + ou - keys to slab along the orthogonal axis')
        print('* Press M to switch between reg_modes (static or dynamic)')
        print('* Press A to start image registration in the selected mode')
        print('* Press C to crop the stack of images given the calculated shifts')
        print('* Press S to save to current stack of image as "input_fname + _REG"')
        print('* Press I to iterate N time over registration and crop in the given mode then save')
        print('--------------------------------------------------\n')
        plt.show()
        
    def filter_keys_event(self,event):
        if event.key in ['+', '-']:
            self.update_slice_index(event)
        elif event.key in ['A']:
            self.align()
        elif event.key in ['S']:
            self.save()
        elif event.key in ['C']:
            self.crop()
        elif event.key in ['I']:
            self.iterate()       
        elif event.key in ['M']:
            self.change_mode()       

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
        self.ax.imshow(self.slice.T, interpolation='none', origin='lower')#, cmap = 'Greys')
        self.fig.canvas.draw()
    
    def progress(self, count, total, start_time):
        try: assert count!=0
        except: count = 0.000001
        bar_len = 50
        filled_len = int(round(bar_len * count / total))
        percents = 100.0 * count / total
        timetaken = time.time() - start_time
        estimated_remaining_time = (timetaken/count) * (total - count) 
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %.1fs%s | %.1fs, %.1fs \r' % (bar, percents, '%', timetaken , estimated_remaining_time))
        if percents >= 100:
            print('[%s] %.1fs%s | %.1fs, %.1fs \r' % (bar, percents, '%', timetaken , estimated_remaining_time))  

        
    def align(self):
        self.shifts = np.zeros((self.data.shape[0],2), dtype = np.int32)
        self.errors = np.zeros((self.data.shape[0]))
        
        print('\n-- Registering volume --')
        print('\nCalculating translational shifts ...')
        st = time.time()    
        if not self.roi:
            self.roi = [0,self.slice.shape[0], 0, self.slice.shape[1]]
            print('full image selected as roi')
        if self.mode == 'S':
            print('reg_mode = static')
            for i, im in enumerate(self.data):
                sh, err, pdiff = register_translation(self.data[self.slice_index, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], im[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], space = 'real')
                self.progress(i+1, self.data.shape[0], st)
                self.shifts[i,:] = sh
                self.errors[i] = err
                
        elif self.mode == 'D':
            print('reg_mode = dynamic')
            for i, im in enumerate(self.data):
                if i == 0:
                    self.shifts[i,:] = [0,0]
                    self.errors[i] = 0
                    continue
                sh, err, pdiff = register_translation(self.data[i-1, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], im[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], space = 'real')
                self.progress(i+1, self.data.shape[0], st)
                self.shifts[i,:] = sh
                self.errors[i] = err
   
        print ('Done.')
        print('resulting shifts #im [sx sy] :')
        fsh = open('Shift.txt', 'w')
        for i,s in enumerate(self.shifts):
            print(i, s)
            fsh.write("%d %d %d\n"%(i, s[0], s[1]))
        fsh.close()
            
        print('\nAligning the stack of images...')
        reg_data = np.zeros_like(self.data)
        st = time.time()  
        for i, im in enumerate(self.data):
            if np.abs(self.shifts[i]).sum()!=0:
                reg_data[i,:,:] = shift(im, self.shifts[i], mode =  'constant' )
            else:
                reg_data[i,:,:] = im
                
            self.progress(i+1, self.data.shape[0], st)
        print ('Done.')
        print('Updating current view to registred data')
        self.data = reg_data
        self.update_figure()
        
    def save(self):
        print('\nSaving resulting volume ...with name %s'%self.output_fname)
        np.save(self.output_fname, np.swapaxes(self.data, 0, self.axis))
        print ('Done.')
        
    def crop(self):
        print('cropping data using min/max shifts as margins')
        
        margin_right = self.shifts[:,0].min()
        margin_left  = self.shifts[:,0].max()
        margin_top    = self.shifts[:,1].min()
        margin_bottom = self.shifts[:,1].max()
        
        print ('right_margin = ',margin_right)
        print ('left_margin = ',margin_left)
        print ('top_margin = ',margin_top)
        print ('bottom_margin = ',margin_bottom)
        
        if margin_right == 0: # if there is no shift at all on one side the range should include all
            margin_right = -1
        if margin_top == 0: # if there is no shift at all on one side the range should include all
            margin_top=-1
        
        self.data = self.data[:,margin_left:margin_right, margin_bottom:margin_top]
        self.update_figure()
        print ('Done.')
        
    def select_roi(self, eclick, erelease):
         'eclick and erelease are the press and release events'
         x1, y1 = int(np.round(eclick.xdata)), int(np.round(eclick.ydata))
         x2, y2 = int(np.round(erelease.xdata)), int(np.round(erelease.ydata))
         y1, y2 = sorted((y1,y2))
         x1, x2 = sorted((x1,x2)) 
         self.ax.add_patch(patches.Rectangle((x1, y1), (x2-x1), (y2-y1), fill=False, linewidth=3))
         self.roi = [y1,y2,x1,x2]
         print('selected roi : ', self.roi)
         plt.draw()

    def iterate(self, N=10):
        print('Starting %d iterations'%N)
        for i in range(N):
            print('iteration #',i)
            self.align()
            self.crop()
            print ('Sum of absolute shifts : ',np.abs(self.shifts).sum(axis = 0))
            if np.abs(self.shifts).sum()==0:
                break
        self.save()
        
    def change_mode(self):
        if self.mode == 'S':
            self.mode = 'D'
            print('Set reg_mode to Dynamic')
        else:
            self.mode = 'S'
            print('Set reg_mode to Static')
        
class main(object):
    def __init__(self, input_fname, axis, init_slice, mode):
        self.data = np.load(input_fname)
        assert self.data.ndim == 3, "data dimensionality should be 3"
        assert mode in ['S', 'D'], "mode should be among 'S' (static) or 'D' (dynamic)"
        assert axis in ['X','Y','Z'], "directional axis should be among 'X', 'Y', 'Z'" 
        self.axis = {'X':0, 'Y':1, 'Z':2}[axis]
        assert init_slice < self.data.shape[self.axis], "slice index should be < to %i for chosen axis %s"%(self.data.shape[self.axis], axis)
        self.slicer = stack_registerer(self.data, self.axis, init_slice, mode, input_fname)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Stack registerer : simple image registration tool''')
    parser.add_argument('-i', action='store', dest='fname',default='', help='input file (standard NumPy binary file format [*.npy]) containing a 3d array')
    parser.add_argument('-a', action='store', dest='axis',default='Z', help="directional axis to slice along, among 'X', 'Y', 'Z' [default=Z]")
    parser.add_argument('-s', action='store', dest='init_slice',default=0, help='initial slice index value along the given axis [default=0]')
    parser.add_argument('-m', action='store', dest='mode',default='S', help='registration mode static (S) using one referenced image, or dynamic (D) one image with the next one [default=S]')
    proxy = parser.parse_args()
    
    assert proxy.fname != '', parser.parse_args(['-h'])
    
    m = main(proxy.fname, proxy.axis.capitalize(), int(proxy.init_slice), proxy.mode.capitalize())  
   


