import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import nibabel as nib
import cv2
import numpy as np
from torch.utils.data import Dataset

# instead of splitting on '/', windows
HERE     = os.path.dirname(os.path.realpath(__file__))     # .../src/datasets
PROJECT  = os.path.abspath(os.path.join(HERE, '..', '..')) # .../project_root
root_dir = PROJECT                                       # project root


# load a NIfTI volume and return as a NumPy array.
def load_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

# extract time index from filename like 'time_012.nii.gz'
def get_time(path: str) -> float:
    basename = os.path.basename(path)
    return float(basename.split('time_')[-1].split('.')[0])

class SyntheticDataset(Dataset):
    def __init__(self,
                base_path: str = 'data/synthesized_3d/',
                subset: str = 'base',
                image_folder: str = 'images',
                mask_folder: str = 'masks',
                target_dim: Tuple[int,int,int] = (256, 256, 256)):

        super().__init__()

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if not os.path.isabs(base_path):
            candidate = os.path.join(project_root, base_path)
        else:
            candidate = base_path

        candidate = os.path.abspath(candidate)
        # if candidate doesn't exist, handle '/src/...' by dropping the 'src' segment. specific to this project structure.
        if not os.path.isdir(candidate):
            parts = candidate.split(os.sep)
            if len(parts) > 2 and parts[1] == 'src':
                remainder = os.sep.join(parts[2:])
                alt = os.path.join(project_root, remainder)
                if os.path.isdir(alt):
                    candidate = alt
        if not os.path.isdir(candidate):
            raise ValueError(f"Base path not found: {base_path}")
        base_path = candidate

        img_root = os.path.join(base_path, subset, image_folder)
        msk_root = os.path.join(base_path, subset, mask_folder)
        if not os.path.isdir(img_root):
            raise ValueError(f"Image root not found: {img_root}")
        if not os.path.isdir(msk_root):
            raise ValueError(f"Mask root not found: {msk_root}")

        # glob sample to confirm structure
        sample_pattern = os.path.join(img_root, 'subject_*', 'time_[0-9][0-9][0-9].nii*')
        if not glob(sample_pattern):
            raise ValueError(f"No timepoint files found under {img_root}")
        self._load = load_volume

        self.image_by_patient: List[List[str]] = []
        self.mask_by_patient:  List[List[str]] = []
        self.max_t: float = 0.0

        for subj_dir in sorted(glob(os.path.join(img_root, 'subject_*'))):
            subj_name = os.path.basename(subj_dir)
            mask_dir = os.path.join(msk_root, subj_name)

            image_paths = sorted(glob(os.path.join(subj_dir, 'time_[0-9][0-9][0-9].nii*')))
            mask_paths  = sorted(glob(os.path.join(mask_dir,  'time_[0-9][0-9][0-9].nii*')))
            if len(image_paths) != len(mask_paths):
                raise AssertionError(f"{subj_name}: {len(image_paths)} images vs {len(mask_paths)} masks")

            self.image_by_patient.append(image_paths)
            self.mask_by_patient.append(mask_paths)
            for p in image_paths:
                self.max_t = max(self.max_t, get_time(p))

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        # returns number of image channels (1 for grayscale)
        return 1


class SyntheticSubset(SyntheticDataset):
    def __init__(self,
                 main_dataset: SyntheticDataset,
                 subset_indices: List[int],
                 return_format: Literal['one_pair','all_pairs','all_subsequences','all_subarrays','full_sequence'],
                 transforms=None,
                 transforms_aug=None):
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
        
        # override to subset of patients
        self.image_by_patient = [main_dataset.image_by_patient[i] for i in subset_indices]
        self.mask_by_patient  = [main_dataset.mask_by_patient[i]  for i in subset_indices]
        self.return_format = return_format
        self.transforms    = transforms
        self.transforms_aug = transforms_aug
        self.max_t = main_dataset.max_t

        # prepare combinations
        self.all_image_pairs = []
        for subj_idx, img_list in enumerate(self.image_by_patient):
            for t0, t1 in itertools.combinations(range(len(img_list)), 2):
                self.all_image_pairs.append((subj_idx, t0, t1))

        self.all_subsequences = []
        self.all_subarrays = []

        for img_list in self.image_by_patient:
            for t0, t1 in itertools.combinations(range(len(img_list)), 2):
                self.all_subarrays.append(img_list[t0:t1+1])

        for img_list in self.image_by_patient:
            # all subsequences of length>=2
            for r in range(2, len(img_list)+1):
                for idxs in itertools.combinations(range(len(img_list)), r):
                    seq = [img_list[i] for i in idxs]
                    self.all_subsequences.append(seq)






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
            image_paths_list = self.image_by_patient[idx]
            mask_paths_list  = self.mask_by_patient[idx]
            t0, t1 = np.random.choice(len(image_paths_list), size=2, replace=False)
            image_paths = [image_paths_list[t0], image_paths_list[t1]]
            mask_paths = [mask_paths_list[t0], mask_paths_list[t1]]
        elif self.return_format == 'all_pairs':
            subj_idx, t0, t1 = self.all_image_pairs[idx]
            image_paths_list = self.image_by_patient[subj_idx]
            mask_paths_list  = self.mask_by_patient[subj_idx]
            image_paths = [image_paths_list[t0], image_paths_list[t1]]
            mask_paths = [mask_paths_list[t0], mask_paths_list[t1]]
        elif self.return_format == 'all_subarrays':
            seq = self.all_subarrays[idx]
            image_paths = seq
            # derive mask_paths from image_paths naming
            mask_paths = [p.replace('.nii.gz','_mask.nii.gz').replace('.nii','_mask.nii') for p in seq]
        elif self.return_format == 'all_subsequences':
            seq = self.all_subsequences[idx]
            image_paths = seq
            mask_paths = [p.replace('.nii.gz','_mask.nii.gz').replace('.nii','_mask.nii') for p in seq]
        elif self.return_format == 'full_sequence':
            image_paths = self.image_by_patient[idx]
            mask_paths  = self.mask_by_patient[idx]
            t0, t1 = None, None

        timestamps = np.array([get_time(p) for p in image_paths], dtype=float)

        if self.return_format in ['one_pair', 'all_pairs']:
            # load volumes and masks
            images = [self._load(p) for p in image_paths]
            masks = [self._load(p) for p in mask_paths]
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
            images = [self._load(p) for p in image_paths]
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
                
        assert image1.ndim == 4 and image1.shape[0] == 1, f"Expected shape (1,H,W,D), got {image1.shape}"

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
        image = self._load(self.image_list[idx])
        mask = self._load(self.mask_list[idx])

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
    if array.ndim == 3:
        return array[None, ...]
    raise ValueError(f"Expected a 3D array, got shape {array.shape}")

def channel_last_to_channel_first(image: np.array) -> np.array:
    if image.ndim == 3:
        return image[None, ...]  # (H, W, D) -> (1, H, W, D)
    elif image.ndim == 4:
        return np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"Unsupported shape {image.shape}")

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    basename = os.path.basename(path)
    time_str = basename.split('time_')[1].split('_mask')[0].split('.')[0]
    return float(time_str)

def normalize_image(image: np.array) -> np.array:
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image * 2 - 1

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