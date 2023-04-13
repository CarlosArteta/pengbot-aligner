import os
import argparse
import yaml
from . import processor


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
    if not os.path.exists(config['images']):
        raise ValueError(f"{config['images']} does not exist")

    if not os.path.exists(config['densities']):
        raise ValueError(f"{config['densities']} does not exist")

    if not os.path.exists(config['nest_diagram']):
        raise ValueError(f"{config['nest_diagram']} does not exist")

    if not os.path.exists(config['nest_reference_image']):
        raise ValueError(f"{config['nest_reference_image']} does not exist")

    if not os.path.exists(config['nest_reference_density']):
        raise ValueError(f"{config['nest_reference_density']} does not exist")

    return config


def main():
    config = parse_cli_args()

    # create object that processes a folder of images for alignment
    folder_processor = processor.FolderProcessor(
        images_dir=config['images'],
        densities_dir=config['densities'],
        fill_missing=config['fill_missing']
    )

    ref_im_unit = folder_processor.im_unit_from_paths(
        im_path=config['nest_reference_image'],
        density_path=config['nest_reference_density'],
        diagram_path=config['nest_diagram']
    )
    folder_processor.add_reference(ref_im_unit)

    print(repr(folder_processor))
    folder_processor.process_folder()


if __name__ == '__main__':
    main()
