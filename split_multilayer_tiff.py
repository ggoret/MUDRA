#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = "Gael Goret"
__copyright__ = "Copyright 2015, CEA"
__version__ = "0.1"
__email__ = "gael.goret@cea.fr"
__status__ = "dev"


from PIL import Image
import sys

class ImageSequence:
    def __init__(self, im):
        self.im = im
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError # end of sequence
            
def main():
    with Image.open(sys.argv[1]) as img:
        i = 0
        for frame in ImageSequence(img):
            print('%s%04d.tif'%(sys.argv[2],i))
            frame.save('%s%04d.tif'%(sys.argv[2],i))
            i += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage : python split_multilayer_tiff.py input_file.tiff output_basename')
    else:
        main()
    sys.exit()
