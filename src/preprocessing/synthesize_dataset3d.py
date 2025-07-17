import os
import numpy as np
from tqdm import tqdm
import nibabel as nib  # for NIfTI I/O
from typing import Tuple

# Generate 3D volumes and masks, doesn't generate in the correct format, will fix

def generate_longitudinal_3d(volume_shape: Tuple[int, int, int] = (64, 64, 64),
                            num_steps: int = 10,
                            initial_radius: Tuple[int, int, int] = (6, 12, 12),
                            final_radius: Tuple[int, int, int] = (18, 24, 24),
                            seed: int = None):  
    """
    Create a time series of 3D volumes and binary masks.
    Returns:
      vols: uint8 array of shape (T, D, H, W)
      masks: bool array of shape (T, D, H, W)
    """
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    D, H, W = volume_shape

    # precompute coordinate grid for ellipsoid math
    z, y, x = np.indices(volume_shape)

    vols = np.zeros((num_steps, D, H, W), dtype=np.uint8)
    masks = np.zeros((num_steps, D, H, W), dtype=bool)

    # random background cuboid
    while True:
        tl = [np.random.randint(0, s // 4) for s in volume_shape]
        br = [np.random.randint(3 * s // 4, s) for s in volume_shape]
        # check that each dimension’s span ≥ 2 * final_radius
        if all((br[i] - tl[i]) >= 2 * final_radius[i] for i in range(3)):
            break

    # random center for the growing ellipsoid, stays within cuboid—final_radius margin
    center = [
        np.random.randint(tl[i] + final_radius[i], br[i] - final_radius[i] + 1)
        for i in range(3)
    ]

    rz = np.linspace(initial_radius[0], final_radius[0], num_steps)
    ry = np.linspace(initial_radius[1], final_radius[1], num_steps)
    rx = np.linspace(initial_radius[2], final_radius[2], num_steps)

    for t in range(num_steps):
        # create a blank volume and fill the cuboid region (background)
        vol = np.zeros(volume_shape, dtype=np.uint8)
        vol[tl[0]:br[0], tl[1]:br[1], tl[2]:br[2]] = 50  # mid-level intensity

        # compute ellipsoid mask and paint foreground
        ell = ((z - center[0]) / rz[t])**2 + \
              ((y - center[1]) / ry[t])**2 + \
              ((x - center[2]) / rx[t])**2 <= 1
        vol[ell] = 200  # high intensity for ellipsoid

        vols[t] = vol
        masks[t] = ell

    return vols, masks


# save generated data as nifti files

def synthesize_dataset_3d(
    out_root: str = None,
    num_subjects: int = 200,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    num_steps: int = 10,
    subset_name: str = "base",):
    """
    Generates and stores a 3D synthetic dataset.
    For each subject, saves one NIfTI volume and mask per timepoint.
    Directory tree:
      <out_root>/base/images/subject_XXXXX/time_000.nii.gz
      <out_root>/base/masks/ subject_XXXXX/time_000.nii.gz
      ...
    """
    # Determine default output folder relative to this script
    if out_root is None:
        here = os.path.dirname(__file__)                             # src/preprocessing
        project_root = os.path.abspath(os.path.join(here, '..', '..'))  # climb to project root
        out_root = os.path.join(project_root, 'data', 'synthesized_3d')

    img_root  = os.path.join(out_root, subset_name, 'images')
    msk_root  = os.path.join(out_root, subset_name, 'masks')

    for subj in tqdm(range(num_subjects), desc="Subjects"):
        vols, masks = generate_longitudinal_3d(
            volume_shape=volume_shape,
            num_steps=num_steps,
            seed=subj
        )

        subj_name    = f"subject_{subj:05d}"
        subj_img_dir = os.path.join(img_root, subj_name)
        subj_msk_dir = os.path.join(msk_root, subj_name)
        os.makedirs(subj_img_dir, exist_ok=True)
        os.makedirs(subj_msk_dir, exist_ok=True)

        for t in range(num_steps):
            img_nii = nib.Nifti1Image(vols[t], affine=np.eye(4))
            msk_nii = nib.Nifti1Image(masks[t].astype(np.uint8), affine=np.eye(4))

            # identical time_* names in both folders
            fname = f"time_{t:03d}.nii.gz"
            nib.save(img_nii, os.path.join(subj_img_dir, fname))
            nib.save(msk_nii, os.path.join(subj_msk_dir, fname))


def main():
    synthesize_dataset_3d(num_subjects=200)


if __name__ == '__main__':
    main()
