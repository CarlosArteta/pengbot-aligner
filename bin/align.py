#!/usr/bin/env python3

import os
import argparse
import yaml
import warnings
from src import processor


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Aligner..'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML file with the required paths'
    )

    config = parser.parse_args()

    with open(config.config, 'r') as config_fp:
        config = yaml.safe_load(config_fp)

    # Validate arguments
    if 'images' not in config:
        raise ValueError("'images' must be specified")

    if not os.path.exists(config['images']):
        raise ValueError(f"{config['images']} does not exist")
    
    if 'densities' not in config and 'locations' not in config:
        raise ValueError("Either 'densities' or 'locations' must be specified")
    
    if 'densities' in config and 'locations' in config:
        raise ValueError("Only one of 'densities' or 'locations' must be specified")

    if 'densities' in config and not os.path.exists(config['densities']):
        raise ValueError(f"{config['densities']} does not exist")

    if 'nest_diagram' not in config:
        raise ValueError("'nest_diagram' must be specified")

    if not os.path.exists(config['nest_diagram']):
        raise ValueError(f"{config['nest_diagram']} does not exist")

    if 'nest_reference_image' not in config:
        raise ValueError("'nest_reference_image' must be specified")
    
    if not os.path.exists(config['nest_reference_image']):
        raise ValueError(f"{config['nest_reference_image']} does not exist")

    if 'nest_reference_density' not in config and 'locations' not in config:
        raise ValueError("Either 'nest_reference_density' or 'locations' must be specified")

    if 'nest_reference_density' in config and not os.path.exists(config['nest_reference_density']):
        raise ValueError(f"{config['nest_reference_density']} does not exist")
    
    if 'fill_missing' not in config:
        warnings.warn("'fill_missing' not specified, defaulting to True", UserWarning)
        config['fill_missing'] = True

    if 'location_mask_size' not in config and 'locations' in config:
        warnings.warn("'location_mask_size' not specified, defaulting to 100", UserWarning)
        config['location_mask_size'] = 100

    return config


def main():
    config = parse_cli_args()

    # create object that processes a folder of images for alignment
    folder_processor = processor.FolderProcessor(
        images_dir=config['images'],
        densities_dir=config['densities'] if 'densities' in config else None,
        locations_path=config['locations'] if 'locations' in config else None,
        bounding_box_size=config['location_mask_size'] if 'location_mask_size' in config else None,
        camera_info_size=config['camera_info_size'] if 'camera_info_size' in config else 50,
        fill_missing=config['fill_missing']
    )

    ref_im_unit = folder_processor.im_unit_from_paths(
        im_path=config['nest_reference_image'],
        density_path=config['nest_reference_density'] if 'nest_reference_density' in config else None,
        diagram_path=config['nest_diagram']
    )
    folder_processor.add_reference(ref_im_unit)

    print(repr(folder_processor))
    folder_processor.process_folder()


if __name__ == '__main__':
    main()
