import os
import cv2
from scipy.io import loadmat

class ImUnit:
    def __init__(
            self, 
            im, 
            density=None, 
            diagram=None, 
            key_points=None, 
            descriptors=None, 
            name=None
        ):
        self.im = im
        self.density = density
        self.diagram = diagram
        self.key_points = key_points
        self.descriptors = descriptors
        self.name = name


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
    
    
class ImCache:
    def __init__(self, cache_size=3):
        self.cache = []
        self.cache_size = cache_size

    def insert(self, im_unit):
        if len(self.cache) >= self.cache_size:
            _ = self.cache.pop(-1)
        self.cache.insert(0, im_unit)

