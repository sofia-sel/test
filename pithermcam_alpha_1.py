# -*- coding: utf-8 -*-
#!/usr/bin/python3
##################################
# MLX90640 Thermal Camera w Raspberry Pi
##################################
import time, board, busio, traceback
import numpy as np
import adafruit_mlx90640
import datetime as dt
import cv2
import logging
import cmapy
from scipy import ndimage

# Set up logging
logging.basicConfig(filename='pithermcam.log', filemode='a',
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(name)s:%(lineno)d] %(message)s',
                    level=logging.WARNING, datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)

class pithermalcam:
    # See https://gitlab.com/cvejarano-oss/cmapy/-/blob/master/docs/colorize_all_examples.md to for options that can be put in this list
    _colormap_list = ['jet', 'bwr', 'seismic', 'coolwarm', 'PiYG_r', 'tab10', 'tab20', 'gnuplot2', 'brg']
    _interpolation_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, 5, 6]
    _interpolation_list_name = ['Nearest', 'Inter Linear', 'Inter Area', 'Inter Cubic', 'Inter Lanczos4', 'Pure Scipy', 'Scipy/CV2 Mixed']
    _current_frame_processed = False  # Tracks if the current processed image matches the current raw image
    i2c = None
    mlx = None
    _temp_min = None
    _temp_max = None
    _raw_image = None
    _image = None
    _file_saved_notification_start = None
    _displaying_onscreen = False
    _exit_requested = False

    def __init__(self, use_f: bool = True, filter_image: bool = False, image_width: int = 1200,
                 image_height: int = 900, output_folder: str = '/home/pi/pithermalcam/saved_snapshots/'):
        self.use_f = use_f
        self.filter_image = filter_image
        self.image_width = image_width
        self.image_height = image_height
        self.output_folder = output_folder

        self._colormap_index = 0
        self._interpolation_index = 3
        self._setup_therm_cam()
        self._t0 = time.time()
        self.update_image_frame()

    def __del__(self):
        logger.debug("ThermalCam Object deleted.")

    def _setup_therm_cam(self):
        """Initialize the thermal camera"""
        try:
            # Initialize I2C bus
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            # Initialize the MLX90640 thermal camera
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
            logger.info("Thermal camera initialized successfully.")
        except Exception as e:
            logger.error("Error initializing thermal camera: %s", e)
            traceback.print_exc()

    def update_image_frame(self):
        """Update the image frame from the thermal camera"""
        try:
            self._raw_image = np.zeros((24, 32), dtype=float)
            self.mlx.getFrame(self._raw_image)
            self._temp_min = np.min(self._raw_image)
            self._temp_max = np.max(self._raw_image)
            self._current_frame_processed = False
        except Exception as e:
            logger.error("Error updating image frame: %s", e)
            traceback.print_exc()

    def get_mean_temp(self):
        """Calculate and return the mean temperature of the image"""
        temp_c = np.mean(self._raw_image)
        temp_f = self._c_to_f(temp_c)
        return temp_c, temp_f

    def _c_to_f(self, c):
        """Convert Celsius to Fahrenheit"""
        return (c * 9 / 5) + 32

    def _process_raw_image(self):
        """Process the raw temp data to a colored image. Filter if necessary"""
        if self._interpolation_index == 5:  # Scale via scipy only - slowest but seems higher quality
            self._image = ndimage.zoom(self._raw_image, 25)  # interpolate with scipy
            self._image = cv2.applyColorMap(self._image, cmapy.cmap(self._colormap_list[self._colormap_index]))
        elif self._interpolation_index == 6:  # Scale partially via scipy and partially via cv2 - mix of speed and quality
            self._image = ndimage.zoom(self._raw_image, 10)  # interpolate with scipy
            self._image = cv2.applyColorMap(self._image, cmapy.cmap(self._colormap_list[self._colormap_index]))
            self._image = cv2.resize(self._image, (800, 600), interpolation=cv2.INTER_CUBIC)
        else:
            self._image = cv2.applyColorMap(self._raw_image, cmapy.cmap(self._colormap_list[self._colormap_index]))
            self._image = cv2.resize(self._image, (800, 600), interpolation=self._interpolation_list[self._interpolation_index])
        
        # Flip the image
        self._image = cv2.flip(self._image, 1)
        
        # Check temperature and apply transparency if necessary
        temp_c, _ = self.get_mean_temp()
        if temp_c < 40:
            # Add alpha channel and set it to zero (fully transparent)
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2BGRA)
            self._image[:, :, 3] = 0
        else:
            # Ensure alpha channel is fully opaque
            if self._image.shape[2] == 4:
                self._image[:, :, 3] = 255
        
        if self.filter_image:
            self._image = cv2.bilateralFilter(self._image, 15, 80, 80)
        self._current_frame_processed = True

    def _add_image_text(self):
        """Set image text content"""
        if self.use_f:
            temp_min = self._c_to_f(self._temp_min)
            temp_max = self._c_to_f(self._temp_max)
            text = f'Tmin={temp_min:+.1f}F - Tmax={temp_max:+.1f}F - FPS={1/(time.time() - self._t0):.1f} - Interpolation: {self._interpolation_list_name[self._interpolation_index]} - Colormap: {self._colormap_list[self._colormap_index]} - Filtered: {self.filter_image}'
        else:
            text = f'Tmin={self._temp_min:+.1f}C - Tmax={self._temp_max:+.1f}C - FPS={1/(time.time() - self._t0):.1f} - Interpolation: {self._interpolation_list_name[self._interpolation_index]} - Colormap: {self._colormap_list[self._colormap_index]} - Filtered: {self.filter_image}'
        
        if self._image.shape[2] == 4:
            cv2.putText(self._image, text, (30, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255, 255), 1)
        else:
            cv2.putText(self._image, text, (30, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        
        self._t0 = time.time()  # Update time to this pull

        # For a brief period after saving, display saved notification
        if self._file_saved_notification_start is not None and (time.monotonic() - self._file_saved_notification_start) < 1:
            cv2.putText(self._image, 'Snapshot Saved!', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255, 255 if self._image.shape[2] == 4 else 255), 2)

    def save_image(self):
        """Save the current image to disk"""
        try:
            filename = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
            filepath = self.output_folder + filename
            cv2.imwrite(filepath, self._image)
            self._file_saved_notification_start = time.monotonic()
            logger.info(f"Image saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            traceback.print_exc()

    def show_image(self):
        """Display the current image on the screen"""
        if not self._current_frame_processed:
            self._process_raw_image()
        self._add_image_text()
        cv2.imshow("Thermal Image", self._image)
        cv2.waitKey(1)
        self._displaying_onscreen = True

    def update(self):
        """Update the thermal camera and process the frame"""
        self.update_image_frame()
        self._process_raw_image()

# Main loop for testing
if __name__ == '__main__':
    cam = pithermalcam(use_f=False, filter_image=True)
    while True:
        cam.update()
        cam.show_image()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

