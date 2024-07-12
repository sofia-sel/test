import cv2
import numpy as np

# Load the images
hd_image = cv2.imread('path_to_hd_image.png', cv2.IMREAD_GRAYSCALE)
thermal_image = cv2.imread('path_to_thermal_image.png', cv2.IMREAD_GRAYSCALE)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(hd_image, None)
keypoints2, descriptors2 = orb.detectAndCompute(thermal_image, None)

# Match features.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches.
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Estimate the transformation matrix.
M, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Decompose the homography to get the scale, offsetX, and offsetY.
sx = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
sy = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
scale_factor = (sx + sy) / 2  # Assuming isotropic scaling
offsetX = M[0, 2]
offsetY = M[1, 2]

print(f'Scale Factor: {scale_factor}')
print(f'Offset X: {offsetX}')
print(f'Offset Y: {offsetY}')

# Warp the thermal image to align with the HD image.
aligned_thermal_image = cv2.warpPerspective(thermal_image, M, (hd_image.shape[1], hd_image.shape[0]))

# Save the aligned image.
cv2.imwrite('aligned_thermal_image.png', aligned_thermal_image)
