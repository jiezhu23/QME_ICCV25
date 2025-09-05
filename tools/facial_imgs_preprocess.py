import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
from tqdm import tqdm
from deepface import DeepFace
from PIL import Image
from retinaface import RetinaFace
from multiprocessing import Pool
import tensorflow as tf
import argparse

tf.config.set_soft_device_placement(True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def get_all_h5_keys(h5file, path='/'):
    """
    Recursively get all leaf keys (deepest keys) in an HDF5 file.
    
    Args:
        h5file (h5py.File): The HDF5 file object.
        path (str): The current path in the HDF5 file, default is the root.
    
    Returns:
        list: A list of all leaf keys (datasets) in the HDF5 file.
    """
    keys = []
    # Traverse the group at the current path
    for key in h5file[path].keys():
        item = h5file[os.path.join(path, key)]
        # If the item is a group, recurse into it
        if isinstance(item, h5py.Group):
            keys.extend(get_all_h5_keys(h5file, os.path.join(path, key)))
        # If the item is a dataset, it is a leaf key
        elif isinstance(item, h5py.Dataset):
            keys.append(os.path.join(path, key))
        else:
            keys = os.path.join(path, key)
    return sorted(keys)

def grab_imgs_path_from_folder(folder_path, img_format='jpg'):
    img_paths = []
    search_pattern = os.path.join(folder_path, '**', f'*.{img_format}')
    img_paths = glob.glob(search_pattern, recursive=True)
    img_paths = sorted(img_paths)
    return img_paths

def combine_h5files(h5file_paths_list, output_h5file):
    """
    Gather all keys from a list of HDF5 files and create a new uniform HDF5 file.
    Args:
        h5file_paths (list): List of paths to the input HDF5 files.
        output_h5file (str): Path to the output HDF5 file.
    """
    all_keys = set()

    # Gather keys from all provided HDF5 files
    for h5file_path in h5file_paths_list:
        with h5py.File(h5file_path, 'r') as h5file:
            keys = get_all_h5_keys(h5file)
            all_keys.update(keys)

    # Create a new HDF5 file and copy the gathered keys
    with h5py.File(output_h5file, 'w') as new_h5file:
        for h5file_path in h5file_paths_list:
            with h5py.File(h5file_path, 'r') as h5file:
                for key in all_keys:
                    if key in h5file:
                        new_h5file.create_dataset(key, data=h5file[key][...])

def visualize_crop_face(img_path):
    try:
        cropped_faces = DeepFace.extract_faces(img_path, detector_backend = 'retinaface', align=False)
    except ValueError:
        cropped_faces = None

    img = cv2.imread(img_path)

    if len(cropped_faces) == 0:
        print('No face detected in the image')
    else:
        for face in cropped_faces:
            # Extract the facial area coordinates
            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def extract_face(img_path):
    # img_path or BGR image array
    faces = RetinaFace.extract_faces(img_path, align = False)
    
    res = []
    for face in faces:
        face = Image.fromarray(face)
        face = face.resize((112, 112))
        res.append(face)
    return res        

def preprocess_facial_imgs_to_h5_from_folder(dataset_path, output_hdf5_file, dataset_folder_name='CCVID'):
    """
    Read image from folder directly.
    """
    imgs_path_list = grab_imgs_path_from_folder(dataset_path)
    print(f'Preprocessing facial dataset from path {dataset_path}...')
    # Create HDF5 file
    with h5py.File(output_hdf5_file, 'a') as h5file:
        for img_path in tqdm(imgs_path_list):
            tmp_path = f"{img_path.split(dataset_folder_name)[-1][:-4]}"
            if tmp_path in h5file:
                continue
            faces = extract_face(img_path)
            if faces:
                for i, face in enumerate(faces):
                    # Create a dataset with the path as the key
                    img_path = img_path.split(dataset_folder_name)[-1]
                    dataset_name = f'{img_path[:-4]}/face_{i}'
                    h5file.create_dataset(dataset_name, data=face)
                    h5file.flush()
            else:
                print(f'No face detected in the image {img_path}')
                
def preprocess_facial_imgs_to_h5_from_h5(f, h5file_keys, output_hdf5_file):
    """
    Read image from preprocessed h5 file.
    """
    # Create HDF5 file
    with h5py.File(output_hdf5_file, 'a') as h5file:
        for img_path in tqdm(h5file_keys):
            # tmp_path = f"{img_path.split(dataset_folder_name)[-1][:-4]}"
            if img_path in h5file:
                continue
            faces = extract_face(f[img_path][:][..., ::-1]) # convert RGB to BGR
            if faces:
                for i, face in enumerate(faces):
                    # Create a dataset with the path as the key
                    dataset_name = f'{img_path}/face_{i}'
                    h5file.create_dataset(dataset_name, data=face)
                    h5file.flush()
            else:
                print(f'No face detected in the image {img_path}')
            
def save_images_to_hdf5(img_paths, h5file, save_prefix):
    """Helper function to save images to the HDF5 file."""
    for img_path in tqdm(img_paths):
        save_name = img_path.split(save_prefix)[-1][:-4]  # Removing file extension
        if save_name not in h5file:
            img = Image.open(img_path)
            h5file.create_dataset(save_name, data=img)
            h5file.flush()

def convert_dataset_to_hdf5(dataset_root, dataset_name):
    assert dataset_name in ['ccvid', 'mevid', 'ltcc'], 'Dataset name must be supported!'
    print(f'Converting original dataset from path {dataset_root}...')
    output_hdf5_file = os.path.join(dataset_root, f'{dataset_name}_enrich.h5')
    # Create HDF5 file
    with h5py.File(output_hdf5_file, 'a') as h5file:
        if dataset_name == 'ccvid':
            imgs_path_list = grab_imgs_path_from_folder(dataset_root)
            save_images_to_hdf5(imgs_path_list, h5file, 'CCVID')

        elif dataset_name == 'mevid':
            train_imgs_path_list = grab_imgs_path_from_folder(os.path.join(dataset_root, 'bbox_train'))
            test_imgs_path_list = grab_imgs_path_from_folder(os.path.join(dataset_root, 'bbox_test'))

            # Save training images
            save_images_to_hdf5(train_imgs_path_list, h5file, 'mevid')
            # Save testing images
            save_images_to_hdf5(test_imgs_path_list, h5file, 'mevid')
        elif dataset_name == 'ltcc':
            train_imgs_path_list = grab_imgs_path_from_folder(os.path.join(dataset_root, 'train'), img_format='png')
            test_imgs_path_list = grab_imgs_path_from_folder(os.path.join(dataset_root, 'test'), img_format='png')
            query_imgs_path_list = grab_imgs_path_from_folder(os.path.join(dataset_root, 'query'), img_format='png')
            save_images_to_hdf5(train_imgs_path_list, h5file, 'LTCC_ReID')
            save_images_to_hdf5(test_imgs_path_list, h5file, 'LTCC_ReID')
            save_images_to_hdf5(query_imgs_path_list, h5file, 'LTCC_ReID')
    return output_hdf5_file

def visualize_images_from_h5(f, selected_keys):
    """
    Visualize images from an HDF5 file in a grid layout.
    
    Parameters:
    - f: h5file object
    - selected_keys: list of keys to visualize  
    """
    num_images_to_visualize = len(selected_keys)

    # Calculate the number of rows and columns for the grid
    num_rows = int(np.ceil(np.sqrt(num_images_to_visualize)))
    num_cols = num_rows  # Set columns equal to rows for a square layout
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    
    # Ensure axs is always a 1D array
    if num_images_to_visualize == 1:
        axs = np.array([axs])  # Wrap in an array to maintain consistency
    else:
        axs = axs.flatten()
    
    # Loop through each key and plot the image
    images = []
    for i, key in enumerate(selected_keys):
        img = f[key][:]
        images.append(img)  # Collect images instead of plotting them

    return images  # Return the list of images instead of plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process facial images and save to HDF5.')
    parser.add_argument('--partition_num', type=int, default=1, help='Number of partitions for processing.')
    parser.add_argument('--idx', type=int, default=0, help='Index of the partition to process.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset.')
    parser.add_argument('--dataset_root', type=str, help='Root path of the dataset.')
    parser.add_argument('--dataset_h5file_path', type=str, help='Path of converted HDF5 dataset.')
    parser.add_argument('--output_root', type=str, help='Type of the dataset.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    
    # convert_dataset_to_hdf5(args.dataset_root, args.dataset_name)
    
    f = h5py.File(args.dataset_h5file_path, 'r')
    keys = get_all_h5_keys(f)
    # divide the keys into partition_num partitions
    # selected_keys = keys[args.idx * len(keys) // args.partition_num: (args.idx + 1) * len(keys) // args.partition_num]
    # preprocess_facial_imgs_to_h5_from_h5(f, selected_keys, os.path.join(args.output_root, f'{args.dataset_name}_face_part_{args.idx}.h5'))
    
    selected_keys = [k for k in keys if 'session4' in k]
    preprocess_facial_imgs_to_h5_from_h5(f, selected_keys, os.path.join(args.output_root, f'{args.dataset_name}_face_enrich.h5'))

    exit()
