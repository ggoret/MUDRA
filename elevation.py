#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "Gael Goret"
__copyright__ = "Copyright 2015, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"

import vtk
from vtk.util import numpy_support as ns
import sys
import numpy as np

class npy_converter(object):
    dtype_dict = {'uint8':vtk.VTK_UNSIGNED_CHAR, 
                  'uint16':vtk.VTK_UNSIGNED_SHORT, 
                  'int8':vtk.VTK_CHAR, 
                  'int16':vtk.VTK_SHORT, 
                  'int32':vtk.VTK_INT,
                  'uint32':vtk.VTK_UNSIGNED_INT, 
                  'float32':vtk.VTK_FLOAT, 
                  'float64':vtk.VTK_DOUBLE}

    def convert(self, data):
        if data.ndim !=2:
            raise Exception('Data dimension should be 2')
        nx = data.shape[0]
        ny = data.shape[1]
        image = vtk.vtkImageData()
        image.SetDimensions(1, ny, nx)
        image.SetExtent(0, 0, 0, ny-1, 0, nx - 1)
        
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            image.SetScalarType(self.dtype_dict[str(data.dtype)]) 
            image.SetNumberOfScalarComponents(1)
        else:
            image.AllocateScalars(self.dtype_dict[str(data.dtype)],1)
        vtk_array = ns.numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=self.dtype_dict[str(data.dtype)])
        image.GetPointData().SetScalars(vtk_array)
        return image
                      
class main(object):
    def __init__(self):
        # Create the RenderWindow, Renderer and both Actors
        self.renderer = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleSwitch())
        self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        # Start by loading some data.
        input_fname = sys.argv[1]
        if len(sys.argv)>2:# selector have been provided
            self.selector = sys.argv[2]
            self.data = np.load(input_fname, mmap_mode='r')
            assert self.select_slice()
        else: # selector have NOT been provided
            self.selector = False
            self.data = np.load(input_fname)
            if not (self.data.ndim == 2):  # If dim not compatible with elevation, ask for a selector
                print('Incompatible data dimentionality : ', self.data.shape)
                self.selector = raw_input('Enter selector to cast input as 2D array (e.g. [42,:,:]) -> ')
                assert self.select_slice()
        
        convertor =  npy_converter()
        self.image =  convertor.convert(self.data)
        
        self.mi, self.ma = self.image.GetScalarRange()
        
        self.warp_factor = 0.
        self.warp_step = 0.001

        geometry = vtk.vtkImageDataGeometryFilter()  
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            geometry.SetInput(self.image)
        else:
            geometry.SetInputData(self.image)
            
        self.warp = vtk.vtkWarpScalar()
        self.warp.SetInputConnection(geometry.GetOutputPort())
        self.warp.SetScaleFactor(1)
        self.warp.UseNormalOn()
        self.warp.SetNormal(1,0,0)
        self.warp.Update()
        
        lut =vtk.vtkLookupTable()
        lut.SetTableRange(self.image.GetScalarRange())
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0.7, 0)
        lut.Build()
        
        merge=vtk.vtkMergeFilter()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            merge.SetGeometry(self.warp.GetOutput())
            merge.SetScalars(self.image)
        else:
            merge.SetGeometryInputData(self.warp.GetOutput())
            merge.SetScalarsData(self.image)
        merge.Update()
        
        self.outline = vtk.vtkOutlineFilter()
        self.outline.SetInputConnection(merge.GetOutputPort())
        self.outline.Update()
        
        outlineMapper = vtk.vtkPolyDataMapper()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            outlineMapper.SetInputConnection(self.outline.GetOutputPort())
        else:
            outlineMapper.SetInputData(self.outline.GetOutputDataObject(0))
        
        box=vtk.vtkActor()
        box.SetMapper(outlineMapper)
        box.GetProperty().SetColor(0,0,0)
        
        self.renderer.AddActor(box)
        
        mapper=vtk.vtkPolyDataMapper()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(self.image.GetScalarRange())
        mapper.SetInputConnection(merge.GetOutputPort())
        
        actor=vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().ShadingOff()
        self.renderer.AddActor(actor)
        
        
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetTitle("")
        scalarBar.SetWidth(0.1)
        scalarBar.SetHeight(0.9)
        scalarBar.SetLookupTable(lut)
        #self.renderer.AddActor2D(scalarBar)
        
        self.build_axes()
        
        self.warp.SetScaleFactor(self.warp_factor)
        self.warp.Update()
        self.outline.Update()
        
        self.renderer.ResetCameraClippingRange()

        self.renderer.SetBackground(1, 1, 1)
        self.renWin.SetSize(500, 500)

        self.camera = self.renderer.GetActiveCamera()
        self.center_on_actor(actor)    
        
        self.iren.AddObserver("CharEvent", self.on_keyboard_input) 
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()
    
    
    def select_slice(self):
        if self.selector:
            try :
                self.data = eval('self.data%s'%self.selector)
            except:
                print('Wrong selector format : %s'%self.selector)
                return False
        if not (self.data.ndim == 2):
            print('Wrong data dimensions : ndim = %d, using selector = %s'%(self.data.ndim, self.selector)) 
            return False
        return True
          
    def center_on_actor(self,actor):
        Xmin,Xmax,Ymin,Ymax,Zmin,Zmax = actor.GetBounds()
        Xavg = (Xmin+Xmax)/2. 
        Yavg = (Ymin+Ymax)/2.
        self.camera.SetFocalPoint(Xavg, Yavg, 0)
        self.camera.SetPosition(Xavg, Yavg, 10*Yavg)
        
    def build_axes(self):
        self.axes = vtk.vtkAxesActor()
        bounds =  self.image.GetBounds()
        shape = [bounds[1],bounds[3], bounds[5]]
        self.axes.SetTotalLength(shape[0], shape[1], shape[2])
        self.axes.SetNormalizedShaftLength( 1, 1, 1 )
        self.axes.SetNormalizedTipLength( 0, 0, 0 )
        self.axes.AxisLabelsOn()
        self.axes.GetZAxisTipProperty().SetColor( 0, 0, 1)
        self.axes.GetZAxisShaftProperty().SetColor( 0, 0, 1)
        self.axes.GetXAxisShaftProperty().SetLineWidth (2)
        self.axes.SetXAxisLabelText('')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        
        self.axes.GetYAxisTipProperty().SetColor( 0, 1, 0)
        self.axes.GetYAxisShaftProperty().SetColor( 0, 1, 0)
        self.axes.GetYAxisShaftProperty().SetLineWidth (2)
        self.axes.SetYAxisLabelText('')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        self.axes.GetXAxisTipProperty().SetColor( 1, 0, 0 )
        self.axes.GetXAxisShaftProperty().SetColor( 1, 0, 0)
        self.axes.GetZAxisShaftProperty().SetLineWidth (2)
        self.axes.SetZAxisLabelText('')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        self.axes.PickableOff()
        self.renderer.AddActor(self.axes)

    def warping(self, key):
        if self.image:
            x,y,_ = self.image.GetSpacing ()
            self.warp
            self.warp_factor = eval('%f %s %f'%(self.warp_factor,key,self.warp_step))
            if self.warp_factor>=0:
                self.axes.SetTotalLength(self.ma*self.warp_factor, (self.data.shape[1]-1)*y, (self.data.shape[0]-1)*x )
            else :
                self.axes.SetTotalLength(self.ma*self.warp_factor, (self.data.shape[1]-1)*y,(self.data.shape[0]-1)*x )
                
            self.warp.SetScaleFactor(self.warp_factor)
            self.warp.Update()
            self.renderer.ResetCameraClippingRange()
            self.outline.Update()
            self.renWin.Render()
            
    def on_keyboard_input(self, obj=None, event=None):
        
        key = self.iren.GetKeyCode()
        if key in ['+', '-']:
            self.warping(key)
    
        
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python elevation.py input_file.npy [selector]')
    else:
        m = main()
    sys.exit()   
   
