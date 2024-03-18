# Author: Ning Wang @2024

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.filters import sato
from skimage import morphology
from scipy import ndimage as ndi
from skimage.measure import regionprops


def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr


def screen(arr):
    height, width = arr.shape[:2]
    center_row = height // 2
    center_col = width // 2
    center = (center_row, center_col)
    radius = 50
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    current_slice = arr[:, :]
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2)
    current_slice_circular = current_slice.copy()
    current_slice_circular[~mask] = 0
    arr[:, :] = current_slice_circular

    return arr


def transpose_data(arr):
    arr_transpose = np.transpose(arr, (2, 1, 0))
    arr_mean = np.mean(arr_transpose, axis=0)
    arr_max = np.max(arr_transpose, axis=0)
    return arr_transpose, arr_mean, arr_max


def run_sato(arr, sigmas):
    black_ridges = False
    mode = 'reflect'
    cval = 0
    filtered_image = sato(arr, sigmas=sigmas, black_ridges=black_ridges, mode=mode, cval=cval)
    return filtered_image


def run_canny(arr, sigma):
    low_threshold = 0.1
    high_threshold = 0.2
    detected_image = canny(arr, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return detected_image


def filter_regions(arr_label):
    regions = regionprops(arr_label)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = width / height
        if aspect_ratio < 0.5:
            arr_label[arr_label == region.label] = 0
    return arr_label


def fill_holes_and_label(arr_mean_canny):
    filled_holes = ndi.binary_fill_holes(arr_mean_canny)
    cleaned_holes = morphology.remove_small_objects(filled_holes, min_size=30)
    labels, num_labels = ndi.label(cleaned_holes)
    return labels, num_labels


def process_left_phase_masks(arr_mean, holes_label_new):
    mask_left = arr_mean * holes_label_new
    left_label, left_number = ndi.label(mask_left)
    mean_values = np.array([
        np.mean(mask_left[left_label == label])
        for label in range(1, left_number + 1)
    ])

    if np.all(np.isnan(mean_values)):
        mask_left = np.zeros_like(mask_left)
    else:
        max_mean_label = np.nanargmax(mean_values) + 1
        mask_left[left_label != max_mean_label] = 0
        mask_left[left_label == max_mean_label] = 1

    return mask_left


def process_right_phase_masks(arr_mean, holes_label_new):
    mask_right = arr_mean * holes_label_new
    label_right, right_number = ndi.label(mask_right)
    mean_values = np.array([
        np.mean(mask_right[label_right == label])
        for label in range(1, right_number + 1)
    ])

    if np.all(np.isnan(mean_values)):
        mask_right = np.zeros_like(mask_right)
    else:
        min_mean_label = np.nanargmin(mean_values) + 1
        mask_right[label_right != min_mean_label] = 0
        mask_right[label_right == min_mean_label] = 1

    return mask_right


def process_masks_by_sato(arr_mean, holes_label_new):
    sato_mask = arr_mean * holes_label_new
    sato_label, sato_number = ndi.label(sato_mask)
    mean_values = np.array([
        np.mean(sato_mask[sato_label == label])
        for label in range(1, sato_number + 1)
    ])

    if np.all(np.isnan(mean_values)):
        sato_mask = np.zeros_like(sato_mask)
    else:
        max_mean_label = np.nanargmax(mean_values) + 1
        sato_mask[sato_label != max_mean_label] = 0
        sato_mask[sato_label == max_mean_label] = 1

    return sato_mask


def find_center_of_mass(mask):
    labeled_mask, num_labels = ndi.label(mask)
    centers = []
    for label in range(1, num_labels + 1):
        labeled_region = labeled_mask == label
        center = ndi.center_of_mass(labeled_region)
        centers.append(center)
    return centers


def find_closest_mask(phase_mask_velocity, sato_mask_velocity):
    # Find center of mass for phase_mask_velocity and sato_mask_velocity
    phase_centers = find_center_of_mass(phase_mask_velocity)
    sato_centers = find_center_of_mass(sato_mask_velocity)

    # Calculate center of whole image
    image_center = np.array(phase_mask_velocity.shape) / 2

    # Calculate distances from image center to centers of masks
    phase_distances = [np.linalg.norm(np.array(center) - image_center) for center in phase_centers]
    sato_distances = [np.linalg.norm(np.array(center) - image_center) for center in sato_centers]

    # Find which center is closer to the center of the whole image
    closest_phase_center = phase_centers[np.argmin(phase_distances)]
    closest_sato_center = sato_centers[np.argmin(sato_distances)]

    # Determine which mask is closer to the image center
    closest_mask = phase_mask_velocity if np.linalg.norm(np.array(closest_phase_center) - image_center) <= np.linalg.norm(np.array(closest_sato_center) - image_center) else sato_mask_velocity

    return closest_mask


def get_curve(phase_path, mask):
    arr = np.load(os.path.join(common_path, phase_path))
    arr = np.transpose(arr, (1, 0, 2))
    blood_velocity_all = []
    blood_flow_rate_all = []

    for slice_index in range(arr.shape[2]):
        arr_phase = arr
        phase_time_mask = arr_phase[:, :, slice_index] * mask
        blood_velocity = []
        blood_flow_rate = []
        PC_labels, label_counts = np.unique(mask, return_counts=True)
        for label, count in zip(np.unique(mask), label_counts):
            if label != 0:
                pixel_values = phase_time_mask[mask == label]
                mean_pixel_value = np.mean(pixel_values)
                velocity_mean = abs((mean_pixel_value / 4096) * 120)  # velocity encode for Siemens
                blood_velocity.append(velocity_mean)
                pixel_number = count
                lumen_area = pixel_number * 0.607 * 0.607
                rbf = abs(velocity_mean * (lumen_area / 100) * 60)  # in ml/min
                blood_flow_rate.append(rbf)
        blood_velocity_all.append(blood_velocity)
        blood_flow_rate_all.append(blood_flow_rate)
        vel = np.concatenate(blood_velocity_all)
        flow = np.concatenate(blood_flow_rate_all)
    return vel, flow


def create_mask_left(magnitude_path, velocity_path, phase_path, sigmas, sigma_canny):
    arr_magnitude = np.load(os.path.join(common_path, magnitude_path))
    arr_velocity = np.load(os.path.join(common_path, velocity_path))
    arr_phase = np.load(os.path.join(common_path, phase_path))

    arr_velocity_20_576_400, arr_velocity_20_576_400_mean, _ = transpose_data(arr_velocity)
    arr_phase_20_576_400, arr_phase_20_576_400_mean, _ = transpose_data(arr_phase)

    arr_velocity_576_20_400 = np.transpose(arr_velocity_20_576_400, (1, 2, 0))
    arr_velocity_sato = run_sato(arr_velocity_576_20_400, sigmas)
    arr_velocity_sato_20_576_400 = np.transpose(arr_velocity_sato, (2, 0, 1))
    arr_sato_velocity_mean = np.mean(arr_velocity_sato_20_576_400, axis=0)
    arr_sato_velocity_mean_screen = screen(arr_sato_velocity_mean)

    arr_sato_velocity_mean_screen_nor = normalize(arr_sato_velocity_mean_screen)
    arr_sato_velocity_mean_canny = run_canny(arr_sato_velocity_mean_screen_nor, sigma_canny)
    arr_sato_velocity_holes_label, arr_sato_velocity_holes_label_number = fill_holes_and_label(arr_sato_velocity_mean_canny)
    arr_sato_velocity_holes_label = filter_regions(arr_sato_velocity_holes_label)
    holes_label_velocity_new = arr_sato_velocity_holes_label
    holes_label_velocity_new, label_velocity_number_new = ndi.label(holes_label_velocity_new)
    phase_mask_velocity = process_left_phase_masks(arr_phase_20_576_400_mean, holes_label_velocity_new)
    sato_mask_velocity = process_masks_by_sato(arr_sato_velocity_mean, holes_label_velocity_new)
    mask = find_closest_mask(phase_mask_velocity, sato_mask_velocity)

    return mask


def create_mask_right(magnitude_path, velocity_path, phase_path, sigmas, sigma_canny):
    arr_magnitude = np.load(os.path.join(common_path, magnitude_path))
    arr_velocity = np.load(os.path.join(common_path, velocity_path))
    arr_phase = np.load(os.path.join(common_path, phase_path))

    arr_velocity_20_576_400, arr_velocity_20_576_400_mean, _ = transpose_data(arr_velocity)
    arr_phase_20_576_400, arr_phase_20_576_400_mean, _ = transpose_data(arr_phase)

    arr_velocity_576_20_400 = np.transpose(arr_velocity_20_576_400, (1, 2, 0))
    arr_velocity_sato = run_sato(arr_velocity_576_20_400, sigmas)
    arr_velocity_sato_20_576_400 = np.transpose(arr_velocity_sato, (2, 0, 1))
    arr_sato_velocity_mean = np.mean(arr_velocity_sato_20_576_400, axis=0)
    arr_sato_velocity_mean_screen = screen(arr_sato_velocity_mean)

    arr_sato_velocity_mean_screen_nor = normalize(arr_sato_velocity_mean_screen)
    arr_sato_velocity_mean_canny = run_canny(arr_sato_velocity_mean_screen_nor, sigma_canny)
    arr_sato_velocity_holes_label, arr_sato_velocity_holes_label_number = fill_holes_and_label(arr_sato_velocity_mean_canny)
    arr_sato_velocity_holes_label = filter_regions(arr_sato_velocity_holes_label)
    holes_label_velocity_new = arr_sato_velocity_holes_label
    holes_label_velocity_new, label_velocity_number_new = ndi.label(holes_label_velocity_new)
    phase_mask_velocity = process_right_phase_masks(arr_phase_20_576_400_mean, holes_label_velocity_new)
    sato_mask_velocity = process_masks_by_sato(arr_sato_velocity_mean, holes_label_velocity_new)
    mask = find_closest_mask(phase_mask_velocity, sato_mask_velocity)

    return mask


# This is the path for patient iBE-3128-007
common_path = "Your path of PC MRI"

# Define sigma values for ridge operator
sigmas_to_sato = (1, 6, 1)

# Define sigma values for Canny edge detector
sigmas_to_canny = 3.0  # sigmas_to_canny = [3.0]

# Function 1 for left main renal artery
mask = create_mask_left('PC_magnitude.npy',
                        'PC_velocity.npy',
                        'PC_phase.npy',
                        sigmas_to_sato,
                        sigmas_to_canny)

# Function 2 for left main renal artery
vel_left, rbf_left = get_curve('PC_RenalArtery_Left_EcgTrig_fb_120_P.npy', mask)





