import os
import aligner
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import cv2
import utils


class FolderProcessor:
    """
    Class to process a folder with images such that a nest diagram is created for each of the images resulting
    from wrapping a given nest diagram using an alignment computed on pairs of images.
    """

    def __init__(
            self,
            images_dir,
            densities_dir,
            im_ext='.JPG',
            density_ext='.mat',
            camera_info_size=50,
            fill_missing=True
    ):
        self.images_dir = images_dir
        self.densities_dir = densities_dir
        self.output_dir = images_dir + '_nest_diagrams'
        self.im_unit_cache = utils.ImCache()
        self.im_ext = im_ext
        self.density_ext = density_ext
        self.images = self.parse_dir(images_dir, ext=im_ext)
        self.densities = self.parse_dir(densities_dir, ext=density_ext)
        self.im_unit_cache = utils.ImCache(cache_size=6)
        self.feature_extractor = aligner.FeatureExtractor(
            im_rescale_factor=0.75,
            camera_info_size=camera_info_size
        )
        self.aligner = aligner.Aligner()
        self.feature_extractor = aligner.FeatureExtractor()
        self.diagrams_to_fill = []
        self.latest_h_matrix = np.nan
        self.fill_missing = fill_missing
        self.filling_tolerance_scale = 0.03
        self.filling_tolerance_translation = 5

    def process_folder(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        for im_name in tqdm(self.images):
            output_fn = im_name.replace(self.im_ext, '.png')
            output_fp = os.path.join(self.output_dir, output_fn)
            target_im_path = os.path.join(self.images_dir, im_name)
            target_density_path = os.path.join(self.densities_dir, im_name.replace(self.im_ext, self.density_ext))

            target_unit = self.im_unit_from_paths(
                im_path=target_im_path,
                density_path=target_density_path,
                diagram_path=''
            )

            target_unit = self.feature_extractor(target_unit)

            h_matrix = None
            for ref_unit in self.im_unit_cache.cache:
                h_matrix = self.aligner(ref_unit=ref_unit, target_unit=target_unit)
                if h_matrix is not None:
                    warped_diagram = self.aligner.warp_diagram(ref_unit.diagram, h_matrix)
                    cv2.imwrite(output_fp, warped_diagram)
                    target_unit.diagram = warped_diagram
                    if self.fill_missing and len(self.diagrams_to_fill) > 0:
                        self.fill_missing_diagrams(h_matrix)
                    # Update latest_h_matrix only after filling missing diagrams
                    self.latest_h_matrix = h_matrix
                    break

            if h_matrix is None:
                print(f'Homography could not be computed for {im_name}')
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
        if (scale_distance < self.filling_tolerance_scale) & \
           (translation_distance < self.filling_tolerance_translation):
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

    def im_unit_from_paths(self, im_path, density_path, diagram_path):
        im_unit = utils.ImUnit(
            im=self.load_image(im_path),
            density=self.load_density(density_path),
            diagram=self.load_image(diagram_path),
            key_points=None,
            descriptors=None,
            name=im_path
        )
        return im_unit

    @staticmethod
    def load_image(im_fp):
        if os.path.exists(im_fp):
            return cv2.imread(im_fp)
        else:
            return None

    @staticmethod
    def load_density(density_fp):
        if os.path.exists(density_fp):
            data = loadmat(density_fp)
            return data['density']
        else:
            return None

    @staticmethod
    def parse_dir(directory, ext):
        """
        Extract list of files with a given extension
        """
        return sorted([f for f in os.listdir(directory) if f.endswith(ext)])

    def __repr__(self):
        r = '\n === Folder Processor === \n'
        r += f'Images: {len(self.images)} in {self.images_dir} \n'
        r += f'Densities: {len(self.densities)} in {self.densities_dir} \n'
        r += f'Image cache size: {self.im_unit_cache.cache_size} image units \n'
        # r += f'Nest diagram: {self.nest_reference_diagram} \n'
        # r += f'Nest reference images: {self.nest_reference_image} \n'
        r += '======================== \n'
        return r



