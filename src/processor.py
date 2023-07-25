import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from . import utils, aligner


class FolderProcessor:
    """
    Class to process a folder with images such that a nest diagram is created for each of the images resulting
    from wrapping a given nest diagram using an alignment computed on pairs of images.
    """

    def __init__(
            self,
            images_dir,
            densities_dir=None,
            locations_path=None,
            im_ext='.JPG',
            density_ext='.mat',
            camera_info_size=50,
            bounding_box_size=100,
            fill_missing=True
    ):
        self.images_dir = images_dir
        self.densities_dir = densities_dir
        self.locations_path = locations_path
        self.output_dir = images_dir.replace('renamed', 'nests')
        self.im_unit_cache = utils.ImCache()
        self.im_ext = im_ext
        self.density_ext = density_ext
        self.images = self.parse_dir(images_dir, ext=im_ext)
        self.densities = self.parse_dir(densities_dir, ext=density_ext) if densities_dir is not None else None
        self.locations = self.parse_locations(locations_path) if locations_path is not None else None
        self.im_unit_cache = utils.ImCache(cache_size=6)
        self.camera_info_size = camera_info_size
        self.bounding_box_size = bounding_box_size
        self.aligner = aligner.Aligner()
        self.feature_extractor = aligner.FeatureExtractor(
            im_rescale_factor=0.75,
            camera_info_size=camera_info_size
        )
        self.diagrams_to_fill = []
        self.latest_h_matrix = None
        self.fill_missing = fill_missing
        self.filling_tolerance_scale = 0.03
        self.filling_tolerance_translation = 5

    def process_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for im_name in tqdm(self.images):
            output_fn = im_name.replace(self.im_ext, '.png')
            output_fp = os.path.join(self.output_dir, output_fn)
            if os.path.exists(output_fp):
                tqdm.write(f'Output for {im_name} exists. Skipping...')
                continue
            target_im_path = os.path.join(self.images_dir, im_name)
            if self.densities_dir is not None:
                target_density_path = os.path.join(self.densities_dir, im_name.replace(self.im_ext, self.density_ext))
            else:
                target_density_path = None

            target_unit = self.im_unit_from_paths(
                im_path=target_im_path,
                density_path=target_density_path if target_density_path is not None else None,
            )

            target_unit = self.feature_extractor(target_unit)

            h_matrix = None
            for ref_unit in self.im_unit_cache.cache:
                h_matrix = self.aligner(ref_unit=ref_unit, target_unit=target_unit)
                if h_matrix is not None:
                    warped_diagram = self.aligner.warp_diagram(ref_unit.diagram, h_matrix)
                    cv2.imwrite(output_fp, warped_diagram)
                    target_unit.diagram = warped_diagram
                    if (self.fill_missing and 
                        len(self.diagrams_to_fill) > 0 and 
                        self.latest_h_matrix is not None):
                        self.fill_missing_diagrams(h_matrix)
                    # Update latest_h_matrix only after filling missing diagrams
                    self.latest_h_matrix = h_matrix
                    break

            if h_matrix is None:
                tqdm.write(f'Homography could not be computed for {im_name}')
                self.diagrams_to_fill.append(im_name)
                continue

            # Update aligner to reduce drifting of the scene
            self.add_reference(target_unit)

    def fill_missing_diagrams(self, h_matrix):
        """
        Produces diagrams for images in a list of diagrams_to_fill if the current and latest
        homography matrices are close enough
        """
        scale_distance = np.linalg.norm(
            h_matrix[[0, 1], [0, 1]] -
            self.latest_h_matrix[[0, 1], [0, 1]]
        )
        translation_distance = np.linalg.norm(
            h_matrix[[0, 1], [2, 2]] -
            self.latest_h_matrix[[0, 1], [2, 2]]
        )
        if (scale_distance < self.filling_tolerance_scale and 
            translation_distance < self.filling_tolerance_translation):
            mean_h_matrix = (h_matrix + self.latest_h_matrix) / 2
            ref_unit = self.im_unit_cache.cache[-1]
            for im_name in self.diagrams_to_fill:
                output_fn = im_name.replace(self.im_ext, '.png')
                output_fp = os.path.join(self.output_dir, output_fn)
                warped_diagram = self.aligner.warp_diagram(ref_unit.diagram, mean_h_matrix)
                cv2.imwrite(output_fp, warped_diagram)
        # Empty diagrams_to_fill regardless
        # If new h_matrix is different, the diagrams cannot be filled
        # If new h_matrix is close, the diagrams will have been filled
        self.diagrams_to_fill = []

    def add_reference(self, im_unit):
        if im_unit.key_points is None or im_unit.descriptors is None:
            im_unit = self.feature_extractor(im_unit)
        self.im_unit_cache.insert(im_unit)

    def im_unit_from_paths(self, im_path, density_path=None, diagram_path=None):
        im_name = os.path.basename(im_path)
        im = utils.load_image(im_path)
        diagram = utils.load_image(diagram_path) if diagram_path is not None else None
        if density_path is not None:
            density = utils.load_density(density_path)
        else:
            # Make mask from locations
            density = utils.make_density_from_locations(
                xy=utils.load_locations(self.locations, im_name),
                im_shape=im.shape[:2],
                bb_size=self.bounding_box_size,
            )

        im_unit = utils.ImUnit(
            im=im, 
            density=density,
            diagram=diagram,
            key_points=None,
            descriptors=None,
            name=im_name,
            path=im_path
        )
        return im_unit

    @staticmethod
    def parse_dir(directory, ext):
        """
        Extract list of files with a given extension
        """
        return sorted([f for f in os.listdir(directory) if f.endswith(ext)])
    
    @staticmethod
    def parse_locations(locations_path):
        """
        Parse CSV with image locations
        """
        locations = pd.read_csv(
            locations_path, 
            usecols=[ 'image_id', 'cluster_x', 'cluster_y']
            )
        return locations

    def __repr__(self):
        r = '\n === Folder Processor === \n'
        r += f'Images: {len(self.images)} in {self.images_dir} \n'
        if self.densities_dir is not None:
            r += f'Densities: {len(self.densities)} in {self.densities_dir} \n'
        if self.locations is not None:
            r += f'Locations: {len(self.locations)} in {self.locations_path} \n'
        r += f'Image cache size: {self.im_unit_cache.cache_size} image units \n'
        # r += f'Nest diagram: {self.nest_reference_diagram} \n'
        # r += f'Nest reference images: {self.nest_reference_image} \n'
        r += '======================== \n'
        return r



