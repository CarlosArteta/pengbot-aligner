import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

class ImUnit:
    def __init__(
            self, 
            im, 
            density=None, 
            diagram=None, 
            key_points=None, 
            descriptors=None, 
            name=None,
            path=None
        ):
        self.im = im
        self.density = density
        self.diagram = diagram
        self.key_points = key_points
        self.descriptors = descriptors
        self.name = name
        self.path = path


def load_image(im_fp):
        if os.path.exists(im_fp):
            return cv2.imread(im_fp)
        else:
            raise FileNotFoundError(f'Image file {im_fp} not found')


def load_density(density_fp):
    if os.path.exists(density_fp):
        data = loadmat(density_fp)
        return data['density']
    else:
        raise FileNotFoundError(f'Density file {density_fp} not found')      
    

def load_locations(locations, image_name):
    """
    Extract list of locations for an image from a pandas dataframe
    """
    locations = locations[locations['image_id'] == image_name]
    locations = locations.dropna()
    if len(locations) == 0:
        xy = np.array([])
    else:
        xy = np.array([
            np.round(locations['cluster_x'].to_numpy()).astype(int), 
            np.round(locations['cluster_y'].to_numpy()).astype(int)
        ]).transpose()
    return xy
    

def make_density_from_locations(xy, im_shape, bb_size=100):  
    """
    Make density mask from xy locations
    """
    im_h, im_w = im_shape[:2]
    density = np.zeros((im_h, im_w)).astype(np.float32)
    half_size = int(bb_size / 2)
    for i in range(len(xy)):
        x = xy[i, 0]
        y = xy[i, 1]
        density[
            np.maximum(y - half_size, 0):np.minimum(y + half_size, im_h), 
            np.maximum(x - half_size, 0):np.minimum(x + half_size, im_w)
        ] = 1 
    return density

    
class ImCache:
    def __init__(self, cache_size=3):
        self.cache = []
        self.cache_size = cache_size

    def insert(self, im_unit):
        if len(self.cache) >= self.cache_size:
            _ = self.cache.pop(-1)
        self.cache.insert(0, im_unit)

