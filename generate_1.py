def generate():
    global hd_output_frame, thermal_output_frame, hd_lock, thermal_lock
    alpha = 0.5  # Transparency factor for blending

    # Define temperature range for the thermal camera
    min_temp = 20.0  # Minimum temperature in °C
    max_temp = 100.0  # Maximum temperature in °C
    threshold_temp = 40.0  # Temperature threshold in °C

    # Calculate pixel value threshold
    threshold_pixel_value = int((threshold_temp - min_temp) * 255 / (max_temp - min_temp))

    while True:
        with hd_lock:
            hd_frame = hd_output_frame.copy() if hd_output_frame is not None else None
        with thermal_lock:
            thermal_frame = thermal_output_frame.copy() if thermal_output_frame is not None else None
        
        if hd_frame is None or thermal_frame is None:
            continue

        # Resize thermal frame to match the HD frame size
        hd_frame = cv2.resize(hd_frame, (640, 480))
        thermal_frame = cv2.resize(thermal_frame, (640, 480))

        # Convert thermal frame to grayscale to process temperature values
        thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Create masks for pixels above and below the threshold
        above_threshold_mask = thermal_gray >= threshold_pixel_value
        below_threshold_mask = ~above_threshold_mask

        # Create 3-channel masks for blending
        above_threshold_mask_3c = np.stack([above_threshold_mask] * 3, axis=-1)
        below_threshold_mask_3c = np.stack([below_threshold_mask] * 3, axis=-1)

        # Apply alpha blending to create translucent effect for pixels above the threshold
        blended_above_threshold = cv2.addWeighted(thermal_frame, alpha, hd_frame, 1 - alpha, 0)

        # Combine the frames using the masks
        blended_frame = np.where(above_threshold_mask_3c, blended_above_threshold, hd_frame)

        # Encode the blended frame
        (flag, encoded_image) = cv2.imencode(".jpg", blended_frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
