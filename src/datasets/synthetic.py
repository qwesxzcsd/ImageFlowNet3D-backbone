import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])

class SyntheticDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/synthesized/',
                 subset: str = 'base',
                 image_folder: str = 'images',
                 mask_folder: str = 'masks',
                 target_dim: Tuple[int] = (256, 256)):
        '''
        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.
        '''
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob(os.path.join(base_path, subset, image_folder, '*')))
        all_mask_folders = sorted(glob(os.path.join(base_path, subset, mask_folder, '*')))

        assert len(all_image_folders) == len(all_mask_folders)

        self.image_by_patient = []
        self.mask_by_patient = []
        self.max_t = 0
        for im_folder in all_image_folders:
            image_paths = sorted(glob('%s/*.png' % im_folder))
            mask_paths = []
            for image_path_ in image_paths:
                mask_path_ = image_path_.replace(image_folder, mask_folder)
                assert os.path.isfile(mask_path_)
                mask_paths.append(mask_path_)
                self.max_t = max(self.max_t, get_time(image_path_))
            self.image_by_patient.append(image_paths)
            self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 3


class SyntheticSubset(SyntheticDataset):

    def __init__(self,
                 main_dataset: SyntheticDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences', 'all_subarrays', 'full_sequence'],
                 transforms = None,
                 transforms_aug = None):
        '''
        A subset of SyntheticDataset.

        In SyntheticDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of SyntheticSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []
        for image_list in self.image_by_patient:
            pair_indices = list(itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])
                self.all_subarrays.append(image_list[idx1 : idx2+1])

            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_patient)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)
        elif self.return_format == 'all_subsequences':
            # If we return all subsequences of images per patient...
            return len(self.all_subsequences)
        elif self.return_format == 'all_subarrays':
            # If we return all subarrays of images per patient...
            return len(self.all_subarrays)
        elif self.return_format == 'full_sequence':
            # If we return the full sequences of images per patient...
            return len(self.image_by_patient)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_patient[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))

            sampled_pair = [
                image_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in sampled_pair
            ])
            timestamps = np.array([get_time(img) for img in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_pair
            ])
            timestamps = np.array([get_time(img) for img in queried_pair])

        elif self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'all_subarrays':
            queried_sequence = self.all_subarrays[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'full_sequence':
            queried_sequence = self.image_by_patient[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        if self.return_format in ['one_pair', 'all_pairs']:
            assert len(images) == 2
            image1, image2 = images[0], images[1]

            if self.transforms is not None:
                transformed = self.transforms(image=image1, image_other=image2)
                image1 = transformed["image"]
                image2 = transformed["image_other"]

            if self.transforms_aug is not None:
                transformed_aug = self.transforms_aug(image=image1)
                image1_aug = transformed_aug["image"]
                image1_aug = normalize_image(image1_aug)
                image1_aug = channel_last_to_channel_first(image1_aug)

            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

            image1 = channel_last_to_channel_first(image1)
            image2 = channel_last_to_channel_first(image2)

            if self.transforms_aug is not None:
                images = np.vstack((image1[None, ...], image2[None, ...], image1_aug[None, ...]))
            else:
                images = np.vstack((image1[None, ...], image2[None, ...]))

        elif self.return_format in ['all_subsequences', 'all_subarrays', 'full_sequence']:
            num_images = len(images)
            assert num_images >= 2
            assert num_images < 10  # NOTE: see `additional_targets` in `transform`.

            # Unpack the subsequence.
            image_list = np.rollaxis(images, axis=0)

            data_dict = {'image': image_list[0]}
            for idx in range(num_images - 1):
                data_dict['image_other%d' % (idx + 1)] = image_list[idx + 1]

            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)

            images = normalize_image(channel_last_to_channel_first(data_dict['image']))[None, ...]
            for idx in range(num_images - 1):
                images = np.vstack((images,
                                    normalize_image(channel_last_to_channel_first(
                                        data_dict['image_other%d' % (idx + 1)]))[None, ...]))

        return images, timestamps

class SyntheticSegSubset(SyntheticDataset):

    def __init__(self,
                 main_dataset: SyntheticDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of SyntheticDataset.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim

        image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]
        mask_by_patient = [
            main_dataset.mask_by_patient[i] for i in subset_indices
        ]

        self.image_list = [image for patient_folder in image_by_patient for image in patient_folder]
        self.mask_list = [mask for patient_folder in mask_by_patient for mask in patient_folder]
        assert len(self.image_list) == len(self.mask_list)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = load_image(self.image_list[idx], target_dim=self.target_dim, normalize=False)
        mask = load_image(self.mask_list[idx], target_dim=self.target_dim, normalize=False, grayscale=True)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = normalize_image(image)
        mask = mask > 128

        image = channel_last_to_channel_first(image)
        mask = add_channel_dim(mask)

        return image, mask
    
def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    # Add the channel dimension to comply with Torch.
    array = array[None, :, :]
    return array

def channel_last_to_channel_first(image: np.array) -> np.array:
    assert len(image.shape) == 3
    image = np.moveaxis(image, -1, 0)
    return image

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = path.split('time_')[1].replace('.png', '')
    # Shall be 3 digits
    assert len(time) == 3
    time = float(time)
    return time

def normalize_image(image: np.array) -> np.array:
    image = (image / 255 * 2) - 1
    return image

def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True, grayscale: bool = False) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        if not grayscale:
            image = np.array(
                cv2.resize(
                    cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                                code=cv2.COLOR_BGR2RGB), target_dim))
        else:
            image = np.array(
                cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        if not grayscale:
            image = np.array(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                            code=cv2.COLOR_BGR2RGB))
        else:
            image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    if normalize:
        image = normalize_image(image)

    # # Channel last to channel first to comply with Torch.
    # image = np.moveaxis(image, -1, 0)

    return image


if __name__ == '__main__':
    dataset = SyntheticDataset(subset='base')
    subset = SyntheticSubset(dataset, subset_indices=[0, 1, 2])
    set_subset = SyntheticSegSubset(dataset, subset_indices=[0, 1, 2])