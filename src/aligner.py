import numpy as np
import cv2
from . import utils


class Aligner:
    """
    Class to align a target pengbot image to a reference image.
    The object is constructed with a reference image, and called with a target image.
    The homography matrix between them is then returned.
    """
    def __init__(
            self,
            keep_descriptors_factor=0.5,
            min_matches=500,
            im_rescale_factor=0.75
    ):
        self.keep_descriptors_factor = keep_descriptors_factor
        self.min_matches = min_matches
        self.im_rescale_factor = im_rescale_factor

        # Output validator
        self.homography_validator = HomographyValidator()

    def __call__(self, ref_unit, target_unit):
        matches = self.match_descriptors(
            target_descriptors=target_unit.descriptors,
            ref_descriptors=ref_unit.descriptors,
            keep_percent=self.keep_descriptors_factor,
            min_matches=self.min_matches
        )
        if matches is None:
            return None

        h_matrix, _ = self.compute_homography(
            target_key_points=target_unit.key_points,
            ref_key_points=ref_unit.key_points,
            matches=matches
        )

        h_matrix = self.rescale_homography(h_matrix, self.im_rescale_factor)

        if self.homography_validator(h_matrix):
            return h_matrix
        else:
            return None

    @staticmethod
    def match_descriptors(target_descriptors, ref_descriptors, keep_percent=0.5, min_matches=500):
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(target_descriptors, ref_descriptors, None)
        if len(matches) < min_matches:
            return None
        matches = sorted(matches, key=lambda x: x.distance)
        # keep only the top matches
        keep = int(max(len(matches) * keep_percent, min_matches))
        matches = matches[:keep]
        return matches

    @staticmethod
    def compute_homography(target_key_points, ref_key_points, matches):
        # allocate memory for (x, y)-coordinates from matches
        pts_target = np.zeros((len(matches), 2), dtype="float")
        pts_ref = np.zeros((len(matches), 2), dtype="float")

        # loop over matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            pts_target[i] = target_key_points[m.queryIdx].pt
            pts_ref[i] = ref_key_points[m.trainIdx].pt

        homography = cv2.findHomography(pts_target, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=2)

        return homography

    @staticmethod
    def rescale_homography(h_matrix, im_scale_factor):
        s_matrix = np.array([
            [1/im_scale_factor, 0, 0],
            [0, 1/im_scale_factor, 0],
            [0, 0, 1]
        ])
        return s_matrix @ h_matrix @ np.linalg.inv(s_matrix)

    @staticmethod
    def warp_diagram(diagram, h_matrix):
        ref_shape = diagram.shape[:2][::-1]
        warped_diagram = cv2.warpPerspective(
            diagram,
            np.linalg.inv(h_matrix),
            ref_shape
        )
        return warped_diagram


class ImPreprocessor:
    """
    Class to preprocess images and density maps.
    It crops the images to remove the in-image camera information,
    and rescales the images a given factor.
    """
    def __init__(self, camera_info_size, im_rescale_factor):
        self.camera_info_size = camera_info_size
        self.im_rescale_factor = im_rescale_factor

    def __call__(self, im):
        im = self.remove_camera_info(im, self.camera_info_size)
        im = self.resize_im(im, self.im_rescale_factor)
        return im

    @staticmethod
    def resize_im(im, rescale_factor):
        width = int(im.shape[1] * rescale_factor)
        height = int(im.shape[0] * rescale_factor)
        dim = (width, height)
        return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def remove_camera_info(im, crop_size):
        height = int(im.shape[0])
        return im[crop_size: height - crop_size, ...]


class HomographyValidator:
    """
    Class to validate that a homography matrix can be used
    or should be discarded. It performs the following checks:

    scale check: Trapped cameras should not be changing scale,
    this scale factors in the h_matrix should be very close to 1
    up to a scaling_tol
    """
    def __init__(self):
        self.scaling_tol = 0.05

    def __call__(self, h_matrix):
        scaling_factors = h_matrix[[0, 1], [0, 1]]
        if any([abs(1 - s) > self.scaling_tol for s in scaling_factors]):
            return False

        return True


class FeatureExtractor:
    def __init__(
            self,
            max_features=2000,
            im_rescale_factor=0.75,
            mask_smoothing_it=5,
            camera_info_size=50,
    ):
        self.max_features = max_features
        self.mask_smoothing_it = mask_smoothing_it
        self.camera_info_size = camera_info_size
        self.im_rescale_factor = im_rescale_factor
        self.im_preprocessor = ImPreprocessor(
            self.camera_info_size,
            self.im_rescale_factor
        )

    def __call__(self, im_unit):
        key_points, descriptors = self.extract_kps_from_im_and_density(
            im=im_unit.im,
            density=im_unit.density
        )
        im_unit.key_points = key_points
        im_unit.descriptors = descriptors
        return im_unit

    def extract_kps_from_im_and_density(self, im, density):
        im = self.im_preprocessor(im)
        density = self.im_preprocessor(density)
        mask = self.get_mask_from_dm(
            density,
            self.mask_smoothing_it
        )
        key_points, descriptors = self.compute_descriptors(
            im=im,
            max_features=self.max_features,
            mask=mask
        )
        return key_points, descriptors

    @staticmethod
    def compute_descriptors(im, max_features, mask):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(max_features)
        (key_points, descriptors) = orb.detectAndCompute(im_gray, mask)
        return key_points, descriptors

    @staticmethod
    def get_mask_from_dm(dm, iterations=5):
        mask = (dm == 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=iterations)
        return 255 * mask
