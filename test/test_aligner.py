import pytest
import pathlib
import numpy as np
from src import utils
from src.aligner import Aligner, FeatureExtractor

PATH = (pathlib.Path(__file__).parent.parent / "test/testdata/").absolute()

def test_aligner_identity():
    src_im_unit = utils.ImUnit(
        im=utils.load_image(str(PATH / 'AITCb2015a_0001.JPG')),
        density=utils.load_density(str(PATH / 'AITCb2015a_0001.mat')),
    )
    identity_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    aligner = Aligner()
    feature_extractor = FeatureExtractor(
            im_rescale_factor=0.75,
            camera_info_size=50
        )
    src_im_unit = feature_extractor(src_im_unit)
    h_matrix = aligner(src_im_unit, src_im_unit)

    np.testing.assert_almost_equal(h_matrix, identity_matrix)