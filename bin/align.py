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

    config_arg = parser.parse_args()

    with open(config_arg.config, 'r') as config_fp:
        config = yaml.safe_load(config_fp)

    # Validate arguments
    if 'root_dir' not in config:
        raise ValueError("'root_dir' must be specified")
    
    if 'camera_name' not in config:
        raise ValueError("'camera_name' must be specified")
    
    if 'camera_collection_id' not in config:
        raise ValueError("'camera_collection_id' must be specified")
    
    root_dir = config['root_dir']
    camera_name = config['camera_name']
    camera_collection_id = config['camera_collection_id']

    images_path = os.path.join(
        root_dir, 
        camera_name, 
        f'{camera_name}_renamed', 
        f'{camera_name}{camera_collection_id}_renamed'
    )

    if not os.path.exists(images_path):
        raise ValueError(f"{images_path} does not exist")

    densities_path = os.path.join(
        root_dir, 
        camera_name, 
        f'{camera_name}_pengbot', 
        f'{camera_name}{camera_collection_id}_pengbot'
    )

    if not os.path.exists(densities_path):
        densities_path = None
        locations_path = os.path.join(
                root_dir, 
                camera_name, 
                f'{camera_name}_metadata', 
                f'{camera_name}{camera_collection_id}_locations.csv'
            )
        if not os.path.exists(locations_path):
            raise ValueError(f"Either {densities_path} or {locations_path} must exist")
        if 'location_mask_size' not in config:
                warnings.warn("'location_mask_size' not specified, defaulting to 100", UserWarning)
                location_mask_size = 100
    else:
        locations_path = None
        location_mask_size = None

    metadata_path = os.path.join(
        root_dir, 
        camera_name, 
        f'{camera_name}_metadata', 
        f'{camera_name}{camera_collection_id}_metadata'
    )

    if 'months_to_process' not in config:
        config['months_to_process'] = []

    months_to_process = config['months_to_process']

    if not os.path.exists(metadata_path) and len(months_to_process) > 0:
        raise ValueError(f"Months {months_to_process} requested but {metadata_path} does not exist")
  
    if 'nest_diagram' not in config:
        raise ValueError("'nest_diagram' must be specified")

    if not os.path.exists(config['nest_diagram']):
        raise ValueError(f"{config['nest_diagram']} does not exist")

    if 'nest_reference_image_id' not in config:
        raise ValueError("'nest_reference_image_id' must be specified")
    
    nest_reference_image_id = config['nest_reference_image_id']
    nest_reference_image_path = os.path.join(
        images_path,
        f'{camera_name}{camera_collection_id}_{nest_reference_image_id}.JPG'
    )
    
    if not os.path.exists(nest_reference_image_path):
        raise ValueError(f"{nest_reference_image_path} does not exist")
   
    if densities_path is not None:
        nest_reference_density_path = os.path.join(
                densities_path,
                f'{camera_name}{camera_collection_id}_{nest_reference_image_id}.mat'
            )
        if not os.path.exists(nest_reference_density_path):
            raise ValueError(f"{nest_reference_density_path} does not exist")
    else:
        nest_reference_density_path = None
    
    if 'fill_missing' not in config:
        warnings.warn("'fill_missing' not specified, defaulting to True", UserWarning)
        config['fill_missing'] = True

    if 'camera_info_size' not in config:
        warnings.warn("'camera_info_size' not specified, defaulting to 50", UserWarning)
        config['camera_info_size'] = 50

    if 'interactive' not in config:
        config['interactive'] = False
    
    config['config_path'] = config_arg.config
    config['images'] = images_path
    config['densities'] = densities_path
    config['locations'] = locations_path
    config['location_mask_size'] = location_mask_size
    config['metadata'] = metadata_path
    config['nest_reference_image'] = nest_reference_image_path
    config['nest_reference_density'] = nest_reference_density_path

    return config


def main():
    config = parse_cli_args()

    # create object that processes a folder of images for alignment
    folder_processor = processor.FolderProcessor(
        images_dir=config['images'],
        densities_dir=config['densities'],
        locations_path=config['locations'],
        bounding_box_size=config['location_mask_size'],
        camera_info_size=config['camera_info_size'],
        fill_missing=config['fill_missing'],
        interactive=config['interactive'],
        config_path=config['config_path']
    )

    ref_im_unit = folder_processor.im_unit_from_paths(
        im_path=config['nest_reference_image'],
        density_path=config['nest_reference_density'] if config['densities'] is not None else None,
        diagram_path=config['nest_diagram']
    )
    folder_processor.add_reference(ref_im_unit)

    print(repr(folder_processor))
    folder_processor.process_folder()


if __name__ == '__main__':
    main()
