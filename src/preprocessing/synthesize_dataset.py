import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import Tuple
from matplotlib import colormaps
from matplotlib import pyplot as plt

def generate_longitudinal(image_shape: Tuple[int] = (256, 256),
                          num_images: int = 10,
                          initial_radius: Tuple[int] = (18, 12),
                          final_radius: Tuple[int] = (36, 48),
                          random_seed: int = None):
    '''
    Generate longitudinal images of an big rectangle containing a small ellipse.
    The big square (eye) remains unchanged, while the small ellipse (geographic atrophy) grows.
    The masks are the binary masks of the small ellipse.
    '''
    
    image_list = [np.zeros((*image_shape, 3), dtype=np.uint8) for _ in range(num_images)]
    mask_list = [np.zeros(image_shape, dtype=np.uint8) for _ in range(num_images)]

    if random_seed is not None:
        np.random.seed(random_seed)

    # First generate the big rectangle.
    square_tl = [int(np.random.uniform(1/8*image_shape[i], 1/4*image_shape[i]/4)) for i in range(2)]
    square_br = [int(np.random.uniform(3/4*image_shape[i], 7/8*image_shape[i])) for i in range(2)]
    square_centroid = np.mean([square_tl, square_br], axis=0)
    for image in image_list:
        # This directly modifies `image_list` since it uses list index reference.
        image[square_tl[0]:square_br[0],
              square_tl[1]:square_br[1], :] = 255
    for i in range(len(image_list)):
        image_list[i] = radially_color_mask_with_colormap(image_list[i])

    # Then generate the increasingly bigger ellipses.
    ellipse_centroid = [int(np.random.uniform(square_tl[i]+final_radius[i],
                                              square_br[i]-final_radius[i])) for i in range(2)]
    radius_x_list = np.linspace(initial_radius[0], final_radius[0], num_images)
    radius_y_list = np.linspace(initial_radius[1], final_radius[1], num_images)
    color_ellipse = np.uint8(np.array(colormaps['hot'](np.random.choice(range(colormaps['hot'].N)))[:3]) * 255)
    for i, image in enumerate(image_list):
        x_arr = np.linspace(0, image_shape[0]-1, image_shape[0])[:, None]
        y_arr = np.linspace(0, image_shape[1]-1, image_shape[1])
        ellipse_mask = ((x_arr-ellipse_centroid[0])/radius_x_list[i])**2 + \
            ((y_arr-ellipse_centroid[1])/radius_y_list[i])**2 <= 1
        # This directly modifies `image_list` since it uses list index reference.
        image[ellipse_mask, :] = color_ellipse
        mask_list[i][ellipse_mask] = 255

    # OpenCV color channel convertion for saving.
    image_list = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in image_list]
    return image_list, mask_list, square_centroid


def radially_color_mask_with_colormap(binary_mask, colormap='twilight', center=None):

    assert binary_mask.shape[2] == 3
    assert (binary_mask[..., 0] == binary_mask[..., 1]).all() and (binary_mask[..., 0] == binary_mask[..., 2]).all()
    binary_mask = binary_mask[..., 0]
    height, width = binary_mask.shape

    if binary_mask.max() > 1:
        assert binary_mask.dtype == np.uint8
        assert binary_mask.max() == 255
        binary_mask = binary_mask > 128

    # Step 1: Determine the center of the mask
    if center is None:
        foreground_arr = np.argwhere(binary_mask)
        center_x, center_y = np.mean(foreground_arr, axis=0).astype(int)
    else:
        center_x, center_y = center

    # Step 2: Create a grid of (x, y) coordinates
    y, x = np.indices((height, width))

    # Step 3: Compute the angular direction (in radians) from the center
    angles = np.arctan2(y - center_y, x - center_x)

    # Normalize angles to range from 0 to 1 (for use with colormap)
    angles_normalized = (angles + np.pi) / (2 * np.pi)

    # Step 4: Apply the colormap to the normalized angles
    colormap_func = plt.get_cmap(colormap)
    colored_angles = colormap_func(angles_normalized)[:, :, :3]

    # Step 5: Color the mask.
    colored_mask = np.zeros((height, width, 3))
    for i in range(3):  # Iterate over the RGB channels
        colored_mask[:, :, i][binary_mask == 1] = colored_angles[:, :, i][binary_mask == 1]

    colored_mask = np.uint8(colored_mask * 255)
    return colored_mask


def synthesize_dataset(save_folder: str = '../../data/synthesized/', num_subjects: int = 200):
    '''
    Synthesize 3 datasets.
    1. The first dataset has no spatial variation. It has pixel-level alignment temporally.
    2. The second dataset has a predictable translation factor.
    3. The third dataset has a predictable rotation factor.
    '''

    for subject_idx in tqdm(range(num_subjects)):
        image_list, mask_list, square_centroid = generate_longitudinal(random_seed=subject_idx)

        # Do nothing.
        dataset = 'base'
        output_image_folder = os.path.join(save_folder, dataset, 'images', f'subject_{str(subject_idx).zfill(5)}')
        output_mask_folder = os.path.join(save_folder, dataset, 'masks', f'subject_{str(subject_idx).zfill(5)}')
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        assert len(image_list) == len(mask_list)
        for time_idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            cv2.imwrite(os.path.join(output_image_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        image)
            cv2.imwrite(os.path.join(output_mask_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        mask)

        # Add translation.
        dataset = 'translation'
        output_image_folder = os.path.join(save_folder, dataset, 'images', f'subject_{str(subject_idx).zfill(5)}')
        output_mask_folder = os.path.join(save_folder, dataset, 'masks', f'subject_{str(subject_idx).zfill(5)}')
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        max_trans_x, max_trans_y = 32, 32
        for time_idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            translation_x = int(2 * max_trans_x / (len(image_list) - 1) * time_idx - max_trans_x)
            translation_y = int(max_trans_y * np.cos(time_idx / len(image_list) * 2*np.pi))
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            image_trans = cv2.warpAffine(image, translation_matrix, (image.shape[0], image.shape[1]))
            mask_trans = cv2.warpAffine(mask, translation_matrix, (mask.shape[0], mask.shape[1]))
            cv2.imwrite(os.path.join(output_image_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        image_trans)
            cv2.imwrite(os.path.join(output_mask_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        mask_trans)

        # Add rotation.
        dataset = 'rotation'
        output_image_folder = os.path.join(save_folder, dataset, 'images', f'subject_{str(subject_idx).zfill(5)}')
        output_mask_folder = os.path.join(save_folder, dataset, 'masks', f'subject_{str(subject_idx).zfill(5)}')
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        for time_idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            angle = np.linspace(0, 180, len(image_list))[time_idx]
            rotation_matrix = cv2.getRotationMatrix2D((square_centroid[1], square_centroid[0]), angle, 1)
            image_rot = cv2.warpAffine(image, rotation_matrix, (image.shape[0], image.shape[1]))
            mask_rot = cv2.warpAffine(mask, rotation_matrix, (mask.shape[0], mask.shape[1]))
            cv2.imwrite(os.path.join(output_image_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        image_rot)
            cv2.imwrite(os.path.join(output_mask_folder,
                                     f'subject_{str(subject_idx).zfill(5)}_time_{str(time_idx).zfill(3)}.png'),
                        mask_rot)
    return


if __name__ == '__main__':
    synthesize_dataset(
        save_folder='../../data/synthesized/',
        num_subjects=200)