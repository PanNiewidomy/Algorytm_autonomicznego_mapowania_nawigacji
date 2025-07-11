import cv2
import numpy as np
import matplotlib.pyplot as plt

# Map parameters
default_map_resolution = 0.05  # m/px

default_threshold = 0.15  # m

default_threshold_px = default_threshold / default_map_resolution

# Cleaning parameters for removing small, random obstacles
free_thresh = 250          # threshold to distinguish free space (white) from obstacles/unknown
median_blur_size = 3       # kernel size for median blur (must be odd)
morph_kernel = np.ones((3, 3), np.uint8)  # kernel for morphological operations
morph_iterations = 1       # number of iterations for morphological ops


def clean_map(map_image, blur_size=median_blur_size):
    """
    Apply median blur to the map to remove small noise (random obstacles).
    """
    return cv2.medianBlur(map_image, blur_size)


def get_distance_transform(map_image,
                           min_distance,
                           map_resolution=default_map_resolution,
                           threshold_px=default_threshold_px):
    """
    Compute the distance transform on the interior free space of the map,
    and highlight candidate positions at approximately `min_distance` from obstacles.

    - Cleans the map to remove small imperfections.
    - Binarizes only true free space (white pixels) and treats unknown/grey as obstacles.
    - Applies morphological closing/opening to further clean the binary map.
    - Calculates distance transform and marks positions at the given distance in red.
    """
    # 1. Clean small noise
    cleaned = clean_map(map_image)



    # 2. Binarize: free space (>= free_thresh) -> 255, else 0
    _, binary = cv2.threshold(cleaned, free_thresh, 255, cv2.THRESH_BINARY)

    # 3. Morphological operations to fill tiny holes and remove small islands
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=morph_iterations)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  morph_kernel, iterations=morph_iterations)

  
    # 4. Compute distance transform on the cleaned binary map
    dist_tf = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 5. Prepare RGB visualization for candidate area
    #    (convert float distance to uint8 for display)
    dist_copy = dist_tf.copy()
    dist_rgb = cv2.cvtColor(dist_copy.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # 6. Highlight pixels at approximately min_distance in red
    target_px = min_distance / map_resolution
    mask_red = np.abs(dist_copy - target_px) < threshold_px
    dist_rgb[mask_red] = (255, 0, 0)
    
    # plt.figure(figsize=(10,4))
    # # Oryginał po lewej
    # plt.subplot(1,2,1)
    # plt.title('Original Map')
    # plt.axis('off')
    # plt.imshow(map_image, cmap='gray')

    # # Oczyszczona po prawej
    # plt.subplot(1,2,2)
    # plt.title('Cleaned Map')
    # plt.axis('off')
    # plt.imshow(cleaned, cmap='gray')

    # plt.tight_layout()
    # plt.show()

    return dist_rgb