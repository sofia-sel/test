import numpy as np
import cv2
import adafruit_mlx90640
import cmapy
import threading
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class pithermalcam:
    def __init__(self, use_f=False, filter_image=True):
        self.use_f = use_f
        self.filter_image = filter_image
        self.mlx = adafruit_mlx90640.MLX90640()
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self._raw_image = np.zeros((24, 32), dtype=np.float32)
        self._image = None

    def update(self):
        """Update the image frame from the thermal camera."""
        try:
            self.mlx.getFrame(self._raw_image)
            self._process_raw_image()
        except Exception as e:
            logger.error("Error updating image frame: %s", e)

    def _process_raw_image(self):
        """Process the raw temperature data to a colored image."""
        normalized_image = np.clip(self._raw_image, 0, 255).astype(np.uint8)
        self._image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        self._image = cv2.resize(self._image, (800, 600), interpolation=cv2.INTER_CUBIC)
        self._image = cv2.flip(self._image, 1)
        if self.filter_image:
            if self._image.shape[2] == 3:
                self._image = cv2.bilateralFilter(self._image, 15, 80, 80)
        temp_c, _ = self.get_mean_temp()
        if temp_c < 40:
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2BGRA)
            self._image[:, :, 3] = 0
        else:
            if self._image.shape[2] == 4:
                self._image[:, :, 3] = 255

    def get_mean_temp(self):
        mean_temp = np.mean(self._raw_image)
        return mean_temp, mean_temp

def capture_thermal_camera():
    cam = pithermalcam(use_f=False, filter_image=True)
    while True:
        cam.update()

if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_thermal_camera)
    capture_thread.start()
