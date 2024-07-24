import numpy as np
import cv2
import adafruit_mlx90640
import cmapy

class ThermalCamera:
    def __init__(self):
        self.mlx = adafruit_mlx90640.MLX90640()
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self._raw_image = np.zeros((24, 32), dtype=np.float32)
        self._image = None

    def update_image_frame(self):
        """Update the image frame from the thermal camera."""
        try:
            self.mlx.getFrame(self._raw_image)
            self._process_raw_image()
        except Exception as e:
            logger.error("Error updating image frame: %s", e)

    def _process_raw_image(self):
        """Process the raw temperature data to a colored image."""
        # Normalize the image to 0-255 range and convert to 8-bit single-channel
        normalized_image = np.clip(self._raw_image, 0, 255).astype(np.uint8)
        
        # Apply colormap
        self._image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        
        # Resize the image if necessary
        self._image = cv2.resize(self._image, (800, 600), interpolation=cv2.INTER_CUBIC)
        
        # Flip the image if needed
        self._image = cv2.flip(self._image, 1)
        
        # Apply bilateral filter if necessary
        if self._image.shape[2] == 3:  # Ensure the image has 3 channels
            self._image = cv2.bilateralFilter(self._image, 15, 80, 80)
        
        # Add alpha channel for transparency if needed
        temp_c, _ = self.get_mean_temp()
        if temp_c < 40:
            # Add alpha channel and set it to transparent
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2BGRA)
            self._image[:, :, 3] = 0
        else:
            # Ensure alpha channel is opaque
            if self._image.shape[2] == 4:
                self._image[:, :, 3] = 255

    def get_mean_temp(self):
        mean_temp = np.mean(self._raw_image)
        return mean_temp, mean_temp  # Placeholder for actual mean temperature calculation

def capture_thermal_camera(cam):
    while True:
        cam.update()

if __name__ == "__main__":
    import threading

    cam = ThermalCamera()
    capture_thread = threading.Thread(target=capture_thermal_camera, args=(cam,))
    capture_thread.start()
