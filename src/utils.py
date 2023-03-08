class ImUnit:
    def __init__(self, im, density, diagram, key_points, descriptors, name):
        self.im = im
        self.density = density
        self.diagram = diagram
        self.key_points = key_points
        self.descriptors = descriptors
        self.name = name


class ImCache:
    def __init__(self, cache_size=3):
        self.cache = []
        self.cache_size = cache_size

    def insert(self, im_unit):
        if len(self.cache) >= self.cache_size:
            _ = self.cache.pop(-1)
        self.cache.insert(0, im_unit)

