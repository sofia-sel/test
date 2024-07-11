import cv2

# Load the HD image and thermal image
hd_image = cv2.imread('path_to_hd_image.jpg')
thermal_image = cv2.imread('path_to_thermal_image.jpg')

# Define the cropping dimensions for the HD image
crop_top = 198
crop_bottom = 152
crop_left = 294
crop_right = 176

# Crop the HD image
cropped_hd_image = hd_image[crop_top:hd_image.shape[0]-crop_bottom, crop_left:hd_image.shape[1]-crop_right]

# Resize the thermal image to match the cropped HD image dimensions
thermal_image_resized = cv2.resize(thermal_image, (cropped_hd_image.shape[1], cropped_hd_image.shape[0]))

# Make the thermal image translucent
alpha = 0.5  # Transparency factor (0.0 to 1.0)
translucent_thermal = cv2.addWeighted(thermal_image_resized, alpha, cropped_hd_image, 1 - alpha, 0)

# Save the resulting image
cv2.imwrite('overlayed_image.jpg', translucent_thermal)

# Display the resulting image
cv2.imshow('Overlayed Image', translucent_thermal)
cv2.waitKey(0)
cv2.destroyAllWindows()
