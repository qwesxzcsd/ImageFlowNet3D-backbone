import matplotlib.pyplot as plt
import nibabel as nib
from typing import Optional, Tuple
import numpy as np

# small helper to visualize 3d images and masks

def visualize_volume(vol, mask=None, slices=None):
    """
    Display 3 orthogonal slices of a 3d volume
    
    args:
        vol: ndarray of shape (D, H, W), the intensity volume
        mask: ndarray of same shape or none, if given
        slices: tuples of ints (z,y,x) slice indices in each mask
            if none, defaults to mid planes
    """
    D, H, W = vol.shape
    #center slices if not provided
    z0, y0, x0 = slices if slices else (D//2, H//2, W//2)

    # get three 2d views
    views = [
        vol[z0, :, :],  # axial
        vol[:, y0, :],  # coronal
        vol[:, :, x0]   # sagittal
    ]
    titles = [f'Axial (z={z0})', f'Coronal (y={y0})', f'Sagittal (x={x0})']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, view, ttl in zip(axes, views, titles):
        ax.imshow(view.T, cmap='gray', origin='lower')
        if mask is not None:
            # pick corresponding mask slice
            m = {
              'Axial':    mask[z0, :, :],
              'Coronal':  mask[:, y0, :],
              'Sagittal': mask[:, :, x0],
            }[ttl.split()[0]]
            ax.contour(m.T, colors='r', linewidths=0.5)
        ax.set_title(ttl)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def visualize_subject_time(subject_folder: str,
                           time_idx: int,
                           slices: Optional[Tuple[int,int,int]] = None):
    """
    Load subject_folder/time_{time_idx:03d}.nii.gz
    and its mask, then call visualize_volume().
    """
    img_path = f"{subject_folder}/time_{time_idx:03d}.nii.gz"
    msk_path = f"{subject_folder}/time_{time_idx:03d}_mask.nii.gz"

    vol = nib.load(img_path).get_fdata().astype(np.uint8)
    mask = nib.load(msk_path).get_fdata().astype(bool)

    visualize_volume(vol, mask=mask, slices=slices)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_folder", help="path to subject_X folder")
    parser.add_argument("time_idx",       type=int, help="time‐step index")
    parser.add_argument("--slices",
                        nargs=3, type=int,
                        help="optional z y x slice indices",
                        default=None)
    args = parser.parse_args()
    visualize_subject_time(
        args.subject_folder,
        args.time_idx,
        tuple(args.slices) if args.slices else None)

