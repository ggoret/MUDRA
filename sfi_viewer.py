#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import with_statement, division, absolute_import, print_function, unicode_literals # ready for the future !

__author__ = "Gael Goret"
__copyright__ = "Copyright 2016, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"

import vtk
from vtk.util import numpy_support as ns

import matplotlib.pyplot as plt

import sys, time
import numpy as np


class npy_reader(object):

    dtype_dict = {'uint8':vtk.VTK_UNSIGNED_CHAR, 
                  'uint16':vtk.VTK_UNSIGNED_SHORT, 
                  'int8':vtk.VTK_CHAR, 
                  'int16':vtk.VTK_SHORT, 
                  'int32':vtk.VTK_INT,
                  'uint32':vtk.VTK_UNSIGNED_INT, 
                  'float32':vtk.VTK_FLOAT, 
                  'float64':vtk.VTK_DOUBLE}

    def read(self, fname):
        data = np.load(fname)

        assert data.ndim ==3, 'Data dimension should be 3'
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        image = vtk.vtkImageData()
        image.SetDimensions(nz,ny,nx)
        image.SetExtent(0, nz-1, 0, ny-1, 0, nx-1)
        
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            image.SetScalarType(self.dtype_dict[str(data.dtype)]) 
            image.SetNumberOfScalarComponents(1)
        else:
            image.AllocateScalars(self.dtype_dict[str(data.dtype)],1)
        vtk_array = ns.numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=self.dtype_dict[str(data.dtype)])
        image.GetPointData().SetScalars(vtk_array)
        del data
        return image


class Interpolator_error(Exception):
    pass

class Interpolator:
    def __init__(self,data):
        self.polydata = vtk.vtkPolyData()
        self.data_source = data.GetOutput()

        probe = vtk.vtkProbeFilter()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            probe.SetInput(self.polydata)
            probe.SetSource(self.data_source)
        else:
            probe.SetInputData(self.polydata) #VTK6
            probe.SetSourceData(self.data_source)

        self.probe = probe

    def interpolate(self, coordinates):
        coords, vtkids = ndarray_to_vtkpoints(coordinates)
        self.polydata.SetPoints(coords)
        del coords
        
        self.probe.Update()
        return numpy_support.vtk_to_numpy(self.probe.GetOutput().GetPointData().GetScalars())
        
    @staticmethod    
    def ndarray_to_vtkpoints(array):
        """Create vtkPoints from double array"""
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(array.shape[0])
        vtkids = {}
        for i in range(array.shape[0]):
            point = array[i]
            vtkid = points.SetPoint(i, point[0], point[1], point[2])
            vtkids[vtkid]=i
        return points,vtkids
    
        
class Interpolator_configurable:
    def __init__(self,data):

        gaussianSmooth = vtk.vtkImageGaussianSmooth()

        gaussianSmooth.SetInputData( data.GetOutput())
        gaussianSmooth.SetDimensionality(1)
        gaussianSmooth.SetStandardDeviation(2.0)
        gaussianSmooth.Update()

        interp = vtk.vtkImageInterpolator()
        interp.Initialize(gaussianSmooth.GetOutput())
        interp.SetOutValue(-42)
        interp.Update()
        self.interp = interp
    
    def set_interpolation_mode(self, mode):
        mode = mode.lower()
        if mode in  ['n','N','nearest']:
            self.interp.SetInterpolationModeToNearest()
        elif mode in  ['c','C', 'cubic']:
            self.interp.SetInterpolationModeToCubic()
        elif mode in  ['l','L', 'linear']:
            self.interp.SetInterpolationModeToLinear()
        else:
            raise Interpolator_error('Wrong interpolation mode')
        self.interp.Update()
    
    def interpolate(self, coordinates):
        return self.interp.Interpolate(coordinates[0], coordinates[1], coordinates[2],0)

class Iso_surface_error(Exception):
    pass

class Isosurface(object):
    def __init__(self, renwin, data, isovalue, color = (0,0.5,0.75), rendering_type = 'surface'):
        self.renwin = renwin
        self.data = data
        self.iren = renwin.GetInteractor()
        self.renderer = renwin.GetRenderers().GetFirstRenderer()
        self.camera = self.renderer.GetActiveCamera()
        
        self.rendering_type = rendering_type # Among 'surface', 'wireframe', 'points'

        self.iso = vtk.vtkMarchingContourFilter()
        self.iso.UseScalarTreeOn()
        self.iso.ComputeNormalsOn()
        
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            self.iso.SetInput(self.data)
        else:
            self.iso.SetInputData(self.data)
        self.iso.SetValue(0,isovalue)

        depthSort = vtk.vtkDepthSortPolyData()
        depthSort.SetInputConnection(self.iso.GetOutputPort())
        depthSort.SetDirectionToBackToFront()
        depthSort.SetVector(1, 1, 1)
        depthSort.SetCamera(self.camera)
        depthSort.SortScalarsOn()
        depthSort.Update()
 
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(depthSort.GetOutputPort())
        mapper.ScalarVisibilityOff()
        mapper.Update()
 
        self.surf = vtk.vtkActor()
        self.surf.SetMapper(mapper)
        self.surf.GetProperty().SetColor(color)
        self.surf.PickableOff()
         
        if self.rendering_type=='wireframe':
            self.surf.GetProperty().SetRepresentationToWireframe()
        elif self.rendering_type=='surface':
            self.surf.GetProperty().SetRepresentationToSurface()
        elif self.rendering_type=='points':
            self.surf.GetProperty().SetRepresentationToPoints() 
            self.surf.GetProperty().SetPointSize(5)
        else:
            self.surf.GetProperty().SetRepresentationToWireframe()
        self.surf.GetProperty().SetInterpolationToGouraud()
        self.surf.GetProperty().SetSpecular(.4)
        self.surf.GetProperty().SetSpecularPower(10)
        
        self.renderer.AddActor(self.surf)
        self.renwin.Render()
        
class Crop_error(Exception):
    pass
    
    
class Crop(vtk.vtkBoxWidget):
    def __init__(self, renwin, volume):
        self.renwin = renwin
        self.iren = renwin.GetInteractor()
        self.mapper = volume.GetMapper()
        self.volume = volume
        self.SetProp3D(volume)
        self.GetOutlineProperty().SetColor(0,0,0)
        self.planes=vtk.vtkPlanes()
        self.SetInteractor(self.iren)
        self.SetPlaceFactor(1)
        self.PlaceWidget()
        self.InsideOutOn()
        self.SetRotationEnabled(0)
        self.GetPlanes(self.planes)
        self.AddObserver("EndInteractionEvent", self.SelectPolygons)
        self.inorout=1
        self.show()
        self.SelectPolygons()

    def SelectPolygons(self, widget=None, event=None):
        (bxmin,bxmax,bymin,bymax,bzmin,bzmax)=self.volume.GetBounds()
        pd = vtk.vtkPolyData()
        self.GetPolyData(pd)
        (xmin,xmax,ymin,ymax,zmin,zmax)=pd.GetBounds()

        tmpxmin=max(bxmin,xmin)
        tmpxmax=min(bxmax,xmax)
        if tmpxmin < tmpxmax :
	        xmin=tmpxmin
	        xmax=tmpxmax
        elif  tmpxmax <= bxmin: #a gauche
	        xmax = bxmin+xmax-xmin
	        xmin = bxmin
        elif  tmpxmin >= bxmax: #a droite
	        xmin = bxmax-xmax+xmin
	        xmax = bxmax
        else : 
            raise Crop_error('uncorrect crop widget dimensions xmin : %s | xmax : %s'%(xmin,xmax))

        tmpymin=max(bymin,ymin)
        tmpymax=min(bymax,ymax)
        if tmpymin < tmpymax :
	        ymin=tmpymin
	        ymax=tmpymax
        elif  tmpymax <= bymin: #a gauche
	        ymax = bymin+ymax-ymin
	        ymin = bymin
        elif  tmpymin >= bymax: #a droite
	        ymin = bymax-ymax+ymin
	        ymax = bymax
        else : 
            raise Crop_error('uncorrect crop widget dimensions ymin : %s | ymax : %s'%(ymin,ymax))
            
        tmpzmin=max(bzmin,zmin)
        tmpzmax=min(bzmax,zmax)
        if tmpzmin < tmpzmax :
	        zmin=tmpzmin
	        zmax=tmpzmax
        elif  tmpzmax <= bzmin: #a gauche
	        zmax = bzmin+zmax-zmin
	        zmin = bzmin
        elif  tmpzmin >= bzmax: #a droite
	        zmin = bzmax-zmax+zmin
	        zmax = bzmax
        else : 
            raise Crop_error('uncorrect crop widget dimensions zmin : %s | zmax : %s'%(zmin,zmax))
        self.PlaceWidget(xmin,xmax,ymin,ymax,zmin,zmax)
        self.GetPlanes(self.planes)
        self.mapper.SetClippingPlanes(self.planes)
        self.renwin.Render()

    def hide(self):
        self.SetEnabled(0)
    
    def show(self):
        self.SetEnabled(1)
        
class scalar_field_interpolator(object):
    def __init__(self, filename = sys.argv[1]):
        # Create the RenderWindow, Renderer and both Actors
        self.renderer = vtk.vtkRenderer()
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.renderer)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renwin)
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleSwitch())
        self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        # Start by loading some data.
        reader = npy_reader()
        self.data = reader.read(filename)
        
        #secondary_data = reader.read(sys.argv[2])
        self.compute_isosurf = False
        
        self.lut = vtk.vtkLookupTable()
        self.lut.SetHueRange(0.7,0)
        self.mi, self.ma = self.data.GetScalarRange()
        self.lut.SetRange(self.mi,self.ma)
        self.lut.Build()
        
        #---------------- scalar field code
        
        # Create transfer mapping scalar value to opacity
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(self.mi, 0)
        opacityTransferFunction.AddPoint(self.ma, 0.4)#TODO

        # Create transfer mapping scalar value to color
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.SetColorSpaceToRGB()
        colorTransferFunction.AddRGBPoint(self.mi, 0, 0, 1)
        colorTransferFunction.AddRGBPoint(self.ma, 1, 0, 0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOff()
        volumeProperty.SetInterpolationTypeToLinear()
         
       
        if vtk.vtkVersion.GetVTKMajorVersion()<6:       
            # The mapper / ray cast function know how to render the data
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
            self.volume_mapper = vtk.vtkVolumeRayCastMapper()
            self.volume_mapper.SetVolumeRayCastFunction(compositeFunction)
            self.volume_mapper.SetInput(self.data)
        else:
            self.volume_mapper = vtk.vtkSmartVolumeMapper()
            self.volume_mapper.SetRequestedRenderModeToGPU()
            self.volume_mapper.SetInputData(self.data) #VTK6+
            # which mode ?
            #self.volume_mapper.SetRequestedRenderModeToDefault()
            #self.volume_mapper.SetRequestedRenderModeToRayCast()

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(volumeProperty)

        #---------------- /endof scalar field code
        
        self.coords = []
        self.colors = []
        self.point_list = []
        self.vtkid = []
        self.sampling_mode = False

        self.picker = vtk.vtkCellPicker()
        
        self.picker.SetTolerance(0.005)
        
        # An outline is shown for context.
        outline = vtk.vtkOutlineFilter()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            outline.SetInput(self.data)
        else:
            outline.SetInputData(self.data) #VTK6
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(outline_mapper)
        self.outline_actor.PickableOff()
        self.outline_actor.GetProperty().SetColor(0,0,0)

        # Add the actors to the renderer, set the background and size
        self.renderer.AddActor(self.outline_actor)
        self.renderer.AddVolume(self.volume)
        self.build_axes()#TODO
        self.renderer.SetBackground(1,1,1)
        self.renwin.SetSize(600, 600)
        
        bounds =  self.data.GetBounds()
        shape = [bounds[1],bounds[3], bounds[5]]
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetFocalPoint(shape[0]/2.,shape[1]/2.,shape[2]/2.)
        self.camera.SetPosition(-shape[0],shape[1]/2.,shape[2]/2.)
        
        self.plane_widget = None
        self.crop_widget = None
        if self.compute_isosurf:
            self.surfaces = []
            self.surfaces.append(Isosurface(self.renwin, data = self.data, isovalue = 32000., color = (0,1,0), rendering_type = 'surface'))
        
        self.iren.Initialize()
        self.renwin.Render()
        
        self.iren.RemoveObservers("CharEvent") 
        self.iren.AddObserver("CharEvent", self.on_keyboard_input)   
        self.iren.Start()
    
    def init_crop_widget(self):
        self.crop_widget = Crop(self.renwin, self.volume)
        
    def init_plane_widget(self):
        self.scalar_bar = vtk.vtkScalarBarActor()
        # Must add this to avoid vtkTextActor error
        self.scalar_bar.SetTitle("Number of counts")
        self.scalar_bar.SetWidth(0.1)
        self.scalar_bar.SetHeight(0.9)
        self.scalar_bar.SetLookupTable(self.lut)
    
        # The image plane widget are used to probe the dataset.
        self.plane_widget = vtk.vtkPlaneWidget()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            self.plane_widget.SetInput(self.data)
        else:
            self.plane_widget.SetInputData(self.data) #VTK6

        self.plane_widget.NormalToXAxisOn()#TODO
        self.plane_widget.SetRepresentationToOutline()
        self.plane_widget.PlaceWidget()
        self.plane_widget.SetResolution(350)#TODO
        
        
        self.plane = vtk.vtkPolyData()
        self.plane_widget.GetPolyData(self.plane)
        
        self.implicit_plane = vtk.vtkPlane()
        self.plane_widget.GetPlane(self.implicit_plane)
        
        self.probe = vtk.vtkProbeFilter()
        if vtk.vtkVersion.GetVTKMajorVersion()<6:
            self.probe.SetInput(self.plane)
            self.probe.SetSource(self.data)
        else:
            self.probe.SetInputData(self.plane) #VTK6
            self.probe.SetSourceData(self.data)
        self.probe.Update()
        
        contour_mapper = vtk.vtkPolyDataMapper()
        contour_mapper.SetInputConnection(self.probe.GetOutputPort())
        contour_mapper.SetScalarRange(self.mi,self.ma)
        contour_mapper.SetLookupTable(self.lut)
        
        self.contour_actor = vtk.vtkActor()
        self.contour_actor.SetMapper(contour_mapper)
        self.contour_actor.GetProperty().ShadingOff()
        self.contour_actor.GetProperty().SetAmbient(0.6)
        self.contour_actor.GetProperty().SetDiffuse(0.4)
        
        
        self.plane_widget.AddObserver('InteractionEvent', self.update_interactive_plane_widget)
        self.plane_widget.AddObserver('StartInteractionEvent', self.on_pick)
        # Associate the widget with the interactor
        self.plane_widget.SetInteractor(self.iren)
        self.plane_widget.SetEnabled(1)
        self.disablation_mode = True
        
        self.renderer.AddActor(self.contour_actor)
        
        self.renderer.AddActor2D(self.scalar_bar)#TODO          
        self.renwin.Render()
    
    def switch_visibility_scalar_field(self):
        self.plane_widget.SetEnabled(not self.plane_widget.GetEnabled())
        self.contour_actor.SetVisibility(self.plane_widget.GetEnabled())   
        self.scalar_bar.SetVisibility(self.plane_widget.GetEnabled()) 
        self.renwin.Render() 
    
        
    def switch_visibility_plane_widget(self):
        self.plane_widget.SetEnabled(not self.plane_widget.GetEnabled())
        self.contour_actor.SetVisibility(self.plane_widget.GetEnabled())   
        self.scalar_bar.SetVisibility(self.plane_widget.GetEnabled()) 
        self.renwin.Render() 
    
    def update_interactive_plane_widget(self, obj=None, event=None):
        self.plane_widget.GetPolyData(self.plane)
        self.probe.Update()
        self.renwin.Render()
    
    def on_keyboard_input(self, obj=None, event=None):
        key = self.iren.GetKeyCode()
        if key in ['!']: # Cropping fields
            if self.crop_widget == None:
                self.init_crop_widget()
            else: 
                self.crop_widget.SetEnabled(not self.crop_widget.GetEnabled())
                
        if key in ['@']: # Sampling points
            self.sampling_mode = not(self.sampling_mode)
            
        if key in ["#"]: #Plane interpolation
            if self.plane_widget == None:
                self.init_plane_widget()
            else:
                self.switch_visibility_plane_widget()
                
        if key in ['d']:
            if self.plane_widget != None:
                self.plane_widget.SetEnabled(not self.plane_widget.GetEnabled())

        if key in ['+']:
            self.plane_widget_translation(1)
        if key in ['-']:
            self.plane_widget_translation(-1)

        if key in ['c']:
            self.volume.SetVisibility(not self.volume.GetVisibility())
            self.renwin.Render()
                
        if key in ["r"]:
            for s in self.point_list[::-1]:
                self.renderer.RemoveActor(s)
                self.point_list.remove(s)
            self.renwin.Render()
            
        if key in ['l']:
            self.get_points_list()
            
        if key in ['$']:
            self.show_plane_data()
            
        if key in ['^']:
            self.screenshot()
    
    def add_sampling_point(self,coord):
        #Create an arrow.
        arrowSource = vtk.vtkArrowSource()
        arrowSource.InvertOn()

        pstart = np.array(self.camera.GetPosition())
        pend = np.array(coord)
 
        # The X axis is a vector from start to end
        norm_vectX = pend - pstart
        norm_vectX = -norm_vectX/np.linalg.norm(norm_vectX)
        
        #The Z axis is an arbitrary vector cross X
        arb_vect =  -1 + np.random.random(3)*2
        norm_vectZ = np.cross(norm_vectX, arb_vect)
        norm_vectZ = norm_vectZ/np.linalg.norm(norm_vectZ)
        
        #The Y axis is Z cross X
        norm_vectY = np.cross(norm_vectZ, norm_vectX)
        matrix = vtk.vtkMatrix4x4()
 
        #Create the direction cosine matrix
        matrix.Identity();
        for i in range(3):
            matrix.SetElement(i, 0, norm_vectX[i])
            matrix.SetElement(i, 1, norm_vectY[i])
            matrix.SetElement(i, 2, norm_vectZ[i])
 
        scale = 20
        transform = vtk.vtkTransform()
        transform.Translate(coord[0], coord[1], coord[2])
        transform.Concatenate(matrix)
        transform.Scale(scale, scale, scale)
        
        #Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        arrow = vtk.vtkActor()
        mapper.SetInputConnection(arrowSource.GetOutputPort())
        arrow.SetUserMatrix(transform.GetMatrix())
        arrow.SetMapper(mapper)
        arrow.GetProperty().SetColor(1,0,0)
        arrow.GetProperty().ShadingOn()
        arrow.GetProperty().SetAmbient(0.6)
        arrow.GetProperty().SetDiffuse(0.4)
                
        self.point_list.append(np.array(coord))
      
        self.renderer.AddActor(arrow)
        
        self.iren.Render()

    def plane_widget_translation(self, t):
        if self.plane_widget is None:
            return
        nrm = np.array(self.plane_widget.GetNormal())
        ctr = np.array(self.plane_widget.GetCenter())

        trans_ctr = ctr + t * nrm
        self.plane_widget.SetCenter(trans_ctr)
        self.plane_widget.UpdatePlacement()
        self.update_interactive_plane_widget()
    
    def get_points_list(self):
        scalars=[]
        coords=[]
        print('-------------------------------------------------------------------')
        print('           coordinates                 :         intensities       ')
        
        for c in self.point_list:
            c = np.floor(c)
            d = self.data.GetPointData().GetScalars().GetTuple1(self.data.FindPoint(c))
            z, y, x = c
            print(x, y, z , '                     : ' , d) #inversion of the x and z axes
            coords.append((x,y,z))
            scalars.append(d)
        print('-------------------------------------------------------------------')

    def on_pick(self, obj=None, evt=None):
        pos = self.iren.GetEventPosition()
        self.picker.AddPickList(self.contour_actor)
        self.picker.PickFromListOn()
        self.picker.Pick(pos[0], pos[1], 0,self.renderer)
        pos = self.picker.GetPickPosition()
        if self.sampling_mode:
            self.add_sampling_point(pos)
            z,y,x = pos
            print('# new coords added : %.2f %.2f %.2f'%(x,y,z)) 
            self.sampling_mode = False
            
    def show_plane_data(self):
        self.probe.Update()
        scalars = self.probe.GetOutput().GetPointData().GetScalars()
        img = ns.vtk_to_numpy(scalars)
        dim = self.plane_widget.GetResolution() + 1
        img = np.reshape(img,(dim,dim))
        plt.imshow(img.T, interpolation='none', origin='lower')
        plt.colorbar()
        plt.show()
        
    def screenshot(self):
        win_to_img_filter =  vtk.vtkWindowToImageFilter()
        win_to_img_filter.SetInput(self.renwin)
        win_to_img_filter.SetMagnification(1)# set the resolution of the output image (3 times the current resolution of vtk render window)
        win_to_img_filter.SetInputBufferTypeToRGBA()# also record the alpha (transparency) channel
        win_to_img_filter.ReadFrontBufferOff()# read from the back buffer
        win_to_img_filter.Update()
 
        writer = vtk.vtkPNGWriter()
        writer.SetFileName("SFI_screenshot_%s.png"%time.strftime("%Y-%m-%d-%H-%M-%S"));
        writer.SetInputConnection(win_to_img_filter.GetOutputPort());
        writer.Write()
        self.renwin.Render()
        
    def build_axes(self):
        self.axes = vtk.vtkAxesActor()
        bounds =  self.data.GetBounds()
        shape = [bounds[1],bounds[3], bounds[5]]
        self.axes.SetTotalLength(shape[0], shape[1], shape[2])
        self.axes.SetNormalizedShaftLength( 1, 1, 1 )
        self.axes.SetNormalizedTipLength( 0, 0, 0 )
        self.axes.AxisLabelsOn()
        self.axes.GetXAxisTipProperty().SetColor( 1, 0, 0 )
        self.axes.GetXAxisShaftProperty().SetColor( 1, 0, 0)
        self.axes.GetXAxisShaftProperty().SetLineWidth (2)
        self.axes.SetXAxisLabelText('z')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        
        self.axes.GetYAxisTipProperty().SetColor( 0, 1, 0)
        self.axes.GetYAxisShaftProperty().SetColor( 0, 1, 0)
        self.axes.GetYAxisShaftProperty().SetLineWidth (2)
        self.axes.SetYAxisLabelText('y')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        self.axes.GetZAxisTipProperty().SetColor( 0, 0, 1)
        self.axes.GetZAxisShaftProperty().SetColor( 0, 0, 1)
        self.axes.GetZAxisShaftProperty().SetLineWidth (2)
        self.axes.SetZAxisLabelText('x')
        txtprop = vtk.vtkTextProperty()
        txtprop.SetColor(0,0,0)
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(50)
        txtprop.SetOpacity(0.5)
        self.axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
        self.axes.PickableOff()
        self.renderer.AddActor(self.axes)   
        self.iren.Render() 


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python scalar_field_interpolator.py input_file.npy')
    else:
        sfi = scalar_field_interpolator()
    sys.exit()

