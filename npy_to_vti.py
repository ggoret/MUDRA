#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

__author__ = "Gael Goret"
__copyright__ = "Copyright 2015, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"

import vtk
from vtk.util import numpy_support as ns
import numpy as np
import time
import sys

dtype_dict = {'uint8':vtk.VTK_UNSIGNED_CHAR, 
              'uint16':vtk.VTK_UNSIGNED_SHORT, 
              'int8':vtk.VTK_CHAR, 
              'int16':vtk.VTK_SHORT, 
              'int32':vtk.VTK_INT,
              'uint32':vtk.VTK_UNSIGNED_INT, 
              'float32':vtk.VTK_FLOAT, 
              'float64':vtk.VTK_DOUBLE}


def fast_array_to_3d_imagedata(data):
    if data.ndim !=3:
        raise Exception('Data dimension should be 3')
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]
    image = vtk.vtkImageData()
    image.SetDimensions(nz,ny,nx)
    image.SetExtent(0, nz-1, 0, ny-1, 0, nx-1)
    
    if vtk.vtkVersion.GetVTKMajorVersion()<6:
        image.SetScalarType(dtype_dict[str(data.dtype)]) 
        image.SetNumberOfScalarComponents(1)
    else:
        image.AllocateScalars(dtype_dict[str(data.dtype)],1)
    vtk_array = ns.numpy_to_vtk(num_array=data.ravel(), deep=False, array_type=dtype_dict[str(data.dtype)])
    image.GetPointData().SetScalars(vtk_array);
    return image
    
def main():
    input_fname = sys.argv[1]
    output_fname = sys.argv[2]
    print('loading data cube')
    data = np.load(input_fname)
    print('data shape : ',data.shape )
    print('min, mean, max : ',data.min(), data.mean(), data.max())
    t = time.time()
    image = fast_array_to_3d_imagedata(data)
    print('vtkImageData Done')
    print('now writing vti file')
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_fname.replace('.vti','')+'.vti')
    if vtk.vtkVersion.GetVTKMajorVersion()<6:
        writer.SetInputConnection(image.GetProducerPort())
    else:
        writer.SetInputData(image) #VTK6
    writer.Update()
    writer.Write()
    
    print('elapsed time : ',time.time() - t, 's')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage : python npy_to_vti.py data_cube.npy output_file.vti')
    else:
        main()
    sys.exit()

