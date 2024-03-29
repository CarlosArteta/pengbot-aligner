import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tkinter as tk
from datetime import datetime
from PIL import ImageTk, Image
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
            metadata_path=None,
            locations_path=None,
            im_ext='.JPG',
            density_ext='.mat',
            camera_info_size=50,
            bounding_box_size=100,
            fill_missing=True,
            interactive=False,
            config_path=None,
            species = '',
            months_to_process = None,
            keep_n_size = 10
    ):
        self.images_dir = images_dir
        self.densities_dir = densities_dir
        self.metadata_path = metadata_path
        self.locations_path = locations_path
        self.output_dir = images_dir.replace("renamed", "nests")
        if species != '':
            self.output_dir = f'{self.output_dir}_{species}'
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
        self.interactive = interactive
        self.alignment_record_path = os.path.join(
            self.output_dir, 
            f'alignment_record_{datetime.today().strftime("%Y-%m-%d")}.csv')
        if os.path.exists(self.alignment_record_path):
            self.alignment_record = pd.read_csv(self.alignment_record_path)
        else:
            self.alignment_record = pd.DataFrame(columns=['image', 'alignment'])
        self.config_path = config_path
        self.months_to_process = None if len(months_to_process) == 0 else months_to_process
        self.keep_n_size = keep_n_size
        self.keep_next_n_count = False

    def process_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.months_to_process is not None:
            metadata = self.get_metadata(self.metadata_path)
            metadata = metadata[metadata['month'].isin(self.months_to_process)]

        for im_name in tqdm(self.images):

            output_fn = im_name.replace(self.im_ext, '.png')
            output_fp = os.path.join(self.output_dir, output_fn)

            if self.keep_next_n_count > 0:
                tqdm.write(f'{im_name} skipped in batch of {self.keep_n_size}')
                cv2.imwrite(output_fp, ref_unit.diagram)
                self.update_alignment_record(im_name, f'keep_next_{self.keep_n_size}')
                self.keep_next_n_count -= 1
                continue

            if self.months_to_process is not None and im_name not in metadata['im_name'].values:
                tqdm.write(f'{im_name} outside months to process. Skipping...')
                continue

            if os.path.exists(output_fp):
                tqdm.write(f'Output for {im_name} exists. Skipping...')
                self.update_alignment_record(im_name, 'existed')
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
                    self.update_alignment_record(im_name, 'auto')
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
                if self.interactive:
                    action = self.get_user_action(os.path.join(self.images_dir, im_name), ref_unit.diagram)
                    if action == 'keep_snow':
                        cv2.imwrite(output_fp, ref_unit.diagram)
                        self.update_alignment_record(im_name, 'keep_snow')
                    elif action == 'keep_other':
                        cv2.imwrite(output_fp, ref_unit.diagram)
                        self.update_alignment_record(im_name, 'keep_other')
                    elif action == 'keep_next_n':
                        cv2.imwrite(output_fp, ref_unit.diagram)
                        self.update_alignment_record(im_name, f'keep_next_{self.keep_n_size}')
                        self.keep_next_n_count = self.keep_n_size - 1
                    elif action == 'ignore_blurry':
                        blank_diagram = np.zeros_like(ref_unit.diagram)
                        cv2.imwrite(output_fp, blank_diagram)
                        self.update_alignment_record(im_name, 'ignore_blurry')
                    elif action == 'ignore_other':
                        blank_diagram = np.zeros_like(ref_unit.diagram)
                        cv2.imwrite(output_fp, blank_diagram)
                        self.update_alignment_record(im_name, 'ignore_other')
                    elif action == 'redraw':
                        redraw_config_path = self.make_redraw_config_file(im_name)
                        print(f'\n Processing stopped to redraw diagram. Config file created at {redraw_config_path.name}')
                        break
                    else:
                        raise ValueError(f'Action {action} not recognized')
                else:
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
     

    def get_user_action(self, im_path, diagram):
        """Get user to input what to do with failed alignments"""
        action = None
        def keep_snow():
            nonlocal action
            action = 'keep_snow'
            root.destroy()
        def keep_other():
            nonlocal action
            action = 'keep_other'
            root.destroy()
        def ignore_blurry():
            nonlocal action
            action = 'ignore_blurry'
            root.destroy()
        def ignore_other():
            nonlocal action
            action = 'ignore_other'
            root.destroy()
        def redraw():
            nonlocal action
            action = 'redraw'
            root.destroy()
        def keep_next_n():
            nonlocal action
            action = 'keep_next_n'
            root.destroy()
        def key_press(event):
            key = event.char
            if key == 's':
                keep_snow()
            elif key == 'o':
                keep_other()
            elif key == 'b':
                ignore_blurry()
            elif key == 'i':
                ignore_other()
            elif key == 'r':
                redraw()
            elif key == 'n':
                keep_next_n()
            else:
                print(f'Key {key} not recognized')             
        
        # load image with opencv
        im = cv2.imread(im_path)
        
        # overlay diagram over image with transparency
        im = cv2.addWeighted(im, 0.75, diagram, 0.25, 0)

        # convert image to RGB format
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # show using tkinter
        root = tk.Tk()
        root.title(im_path)
        resize_factor = root.winfo_screenwidth() * 0.6 / im.shape[1]
        im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)
        im = Image.fromarray(im)
        img = ImageTk.PhotoImage(image=im)
        panel = tk.Label(root, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        # Add buttons 
        button_frame = tk.Frame(root)
        button_frame.pack(side="top", fill="both", expand="yes")
        
        keep_snow_button = tk.Button(button_frame, text="(s) Keep diagram (snow)", command=keep_snow)
        keep_snow_button.pack(side="left", fill="both", expand="yes")
        keep_other_button = tk.Button(button_frame, text="(o) Keep diagram (other)", command=keep_other)
        keep_other_button.pack(side="left", fill="both", expand="yes")
        keep_next_n_button = tk.Button(button_frame, text=f"(n) Keep next {self.keep_n_size}", command=keep_next_n)
        keep_next_n_button.pack(side="left", fill="both", expand="yes")
        ignore_blurry_button = tk.Button(button_frame, text="(b) Ignore (blurry)", command=ignore_blurry)
        ignore_blurry_button.pack(side="left", fill="both", expand="yes")
        ignore_other_button = tk.Button(button_frame, text="(i) Ignore (other)", command=ignore_other)
        ignore_other_button.pack(side="left", fill="both", expand="yes")
        redraw_button = tk.Button(button_frame, text="(r) Redraw", command=redraw)
        redraw_button.pack(side="left", fill="both", expand="yes")
        
        root.bind('<Key>', key_press) 

        # Display
        root.mainloop()

        return action
    
    def update_alignment_record(self, im_name, alignment):
        if alignment == 'existed' and im_name in self.alignment_record['image'].values:
            return
        else:
            self.alignment_record = pd.concat(
                [self.alignment_record, pd.DataFrame({'image': im_name, 'alignment': alignment}, index=[0])], 
                ).reset_index(drop=True)
            self.alignment_record.to_csv(self.alignment_record_path, index=False)

    def make_redraw_config_file(self, im_name):
        """
        Make config file to redraw a diagram
        """
        redraw_config_path = os.path.join(os.path.dirname(self.config_path), f'config_{im_name.replace(self.im_ext, ".yaml")}')
        if os.path.exists(redraw_config_path):
            raise ValueError(f'Config file {redraw_config_path} already exists')
        else:
            with open(self.config_path, 'r') as config_path:
                config = yaml.safe_load(config_path)
            config['nest_diagram'] = os.path.join(self.output_dir, im_name.replace(self.im_ext, '.png'))
            config['nest_reference_image_id'] = im_name.replace(self.im_ext, '').split('_')[-1]
            with open(redraw_config_path, 'w') as redraw_config_path:
                yaml.dump(config, redraw_config_path)
        return redraw_config_path



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
    
    @staticmethod
    def get_metadata(metadata_path):
        """
        Parse metadata file
        """
        metadata = pd.read_csv(metadata_path)
        metadata['date'] = pd.to_datetime(metadata['datetime'], format='%Y:%m:%d %H:%M:%S')
        metadata['month'] = metadata['date'].dt.month
        metadata['im_name'] = metadata['imageid'].apply(lambda x: x.split('/')[-1])
        return metadata

    def __repr__(self):
        r = '\n === Folder Processor === \n'
        r += f'Images: {len(self.images)} in {self.images_dir} \n'
        if self.densities_dir is not None:
            r += f'Densities: {len(self.densities)} in {self.densities_dir} \n'
        if self.locations is not None:
            r += f'Locations: {len(self.locations)} in {self.locations_path} \n'
        r += f'Output: {self.output_dir} \n'
        r += f'Image cache size: {self.im_unit_cache.cache_size} image units \n'
        # r += f'Nest diagram: {self.nest_reference_diagram} \n'
        # r += f'Nest reference images: {self.nest_reference_image} \n'
        r += '======================== \n'
        return r



