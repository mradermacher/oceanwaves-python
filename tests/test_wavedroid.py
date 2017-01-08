from nose.tools import *
import numpy as np
import oceanwaves
import os

fn = os.path.join('..\data\wavedroid\WDwebdownload_dummy.csv')
wd = oceanwaves.WaveDroidReader()
wd.readfile(fn)
ow = wd.to_oceanwaves()