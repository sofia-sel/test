import cv2
import numpy as np

# Load images
rggb_image = cv2.imread('specs.png')
thermal_image = cv2.imread('face.png')

# Convert RGGB to RGB (if needed)
# Assuming 'rggb_image' is already in RGB format

# Resize thermal image for initial guess
resized_thermal = cv2.resize(thermal_image, (rggb_image.shape[1], rggb_image.shape[0]))

# Convert to grayscale
gray_rggb = cv2.cvtColor(rggb_image, cv2.COLOR_BGR2GRAY)
gray_thermal = cv2.cvtColor(resized_thermal, cv2.COLOR_BGR2GRAY)

# Feature detection using ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray_rggb, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_thermal, None)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches (optional)
matched_img = cv2.drawMatches(rggb_image, keypoints1, resized_thermal, keypoints2, matches[:10], None, flags=2)
cv2.imshow("Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find transformation matrix using RANSAC
matrix, mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)

# Apply transformation to the thermal image
aligned_thermal = cv2.warpAffine(thermal_image, matrix, (rggb_image.shape[1], rggb_image.shape[0]))

# Display aligned image
cv2.imshow("Aligned Thermal", aligned_thermal)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate scale factor, offsetX, and offsetY
scale_x = np.linalg.norm(matrix[0, :2])
scale_y = np.linalg.norm(matrix[1, :2])
offset_x = matrix[0, 2]
offset_y = matrix[1, 2]

print(f'Scale factor: {scale_x}, {scale_y}')
print(f'Offset X: {offset_x}')
print(f'Offset Y: {offset_y}')
