"""
Hold and process radar observation data.
"""
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import cv2

from weather_alert.coord_helper import get_destination_point
from weather_alert.globalmaptiles import GlobalMercator
from weather_alert.values import Values

VALUES = Values()


def cut_low_rain(img: np.ndarray, thresh: int, max_val: int) -> np.ndarray:
    """
    Remove low rain areas (clouds) from the image.

    :param img: np array img
    :param thresh: threshold
    :param max_val: max value
    :return: np array img filtered
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_clean = cv2.bitwise_and(img, mask_rgb)
    return img_clean


def add_alpha_channel(img: np.ndarray, thresh: int, max_val: int) -> np.ndarray:
    """
    Add alpha channel to image.

    :param img: np array img
    :param thresh: threshold
    :param max_val: max value
    :return: np array img with alpha channel
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY)
    alpha_channel = cv2.bitwise_not(mask)
    bgr = cv2.split(img)
    bgra = list(bgr) + [alpha_channel]
    img_alpha = cv2.merge(bgra)
    return img_alpha


def draw_frame(img: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    Draw frames on image.

    :param img: source np array img as background for frames
    :param frame: np array img with frames and annotations
    :return: np array img with frames
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.bitwise_and(frame, frame, mask=mask)
    map_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img_added = cv2.add(map_bg, img_fg)
    return img_added


def blend_map_rain_light(img: np.ndarray, img_rain: np.ndarray, map_tile: np.ndarray) -> np.ndarray:
    """
    Blend map and rain with light map.

    :param img: np array img non blured rain layer for mask
    :param img_rain: np array img with blured rain layer as foreground
    :param map_tile: np array img with map as background
    :return: np array img blended
    """
    gray = cv2.cvtColor(img_rain, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    map_bg = cv2.bitwise_and(map_tile, map_tile, mask=mask_inv)
    img_blended = cv2.add(map_bg, img_fg)
    return img_blended


def blend_map_rain(img: np.ndarray, map_tile: np.ndarray) -> np.ndarray:
    """
    Blend map and rain with darken map.

    :param img: np array img of rain layer as foreground
    :param map_tile: np array img with map as background
    :return: np array img blended
    """
    alpha = 0.3
    img_blended = cv2.addWeighted(map_tile, alpha, img, 1.0 - alpha, 0.5)
    return img_blended


def add_map_rain(img: np.ndarray, map_tile: np.ndarray) -> np.ndarray:
    """
    Add map and rain. No transparency of rain layer.

    :param img: np array img of rain layer
    :param map_tile: np array img with map
    :return: np array img added map and rain
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    map_bg = cv2.bitwise_and(map_tile, map_tile, mask=mask_inv)
    img_added = cv2.add(map_bg, img_fg)
    return img_added


def blur_rain_light(img: np.ndarray) -> np.ndarray:
    """
    Blur with white background.

    :param img: np array img
    :return: np array img blured
    """
    img_blur_light = np.copy(img)
    mask = cv2.inRange(img_blur_light, np.array([0, 0, 0]), np.array([0, 0, 0]))
    img_blur_light[mask > 0] = (200, 200, 200)
    img_blur_light = cv2.GaussianBlur(img_blur_light, (3, 3), 0)
    mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([0, 0, 0]))
    img_blur_light[mask > 0] = (0, 0, 0)
    return img_blur_light


def blur_rain(img: np.ndarray) -> np.ndarray:
    """
    Blur with black background.

    :param img: np array img
    :return: np array img blured
    """
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    return img_blur


def detect_rain(img_origo: np.ndarray, top: int, bottom: int, left: int,
                right: int, frame: dict, status: str) -> [float, str]:
    """
    Detect rain in the frame.

    :param img_origo: np array img
    :param top: top position of frame in pixels
    :param bottom: bottom position of frame in pixels
    :param left: left position of frame in pixels
    :param right: right position of frame in pixels
    :param frame: frame size and type
    :param status: initial status [OK] should be updated to
    [WARNING], [ALERT] or [RAIN] if rain detected
    :return: percentage of rain, status of rain of desired frame.
    """
    img_crop = img_origo[top:bottom, left:right]
    color = np.array(VALUES.color_black, dtype=np.uint8)
    mask = cv2.inRange(img_crop, color, color)
    cv2.bitwise_and(img_crop, img_crop, mask=mask)
    ratio = cv2.countNonZero(mask) / (img_crop.size / 3)
    percentage = 100 - np.round(ratio * 100, 2)
    if percentage > 0:
        status = frame['frame']
    return percentage, status


class Observation:
    """
    Class representing radar observation.
    """

    def __init__(self,
                 path: str,
                 timestamp: int,
                 lat: float,
                 lon: float,
                 radar_config: dict,
                 observation_type: str,
                 zoom: int,
                 map_tile: np.ndarray):

        self.time = int(timestamp)
        self.date_str = datetime.fromtimestamp(self.time)
        self.path = path
        self.type = observation_type
        self.latitude = lat
        self.longitude = lon
        self.radar_config = radar_config
        self.img = None
        self.img_annotated = None
        self.img_annotation = None
        self.map_tile = map_tile
        self.status = VALUES.status_ok
        self.status_percentage = 0
        self.status_percentage_warn = 0
        self.status_percentage_alert = 0
        self.status_percentage_rain = 0
        self.size = int(self.radar_config['size'])  # image size, can be 256 or 512
        self.zoom = zoom  # map zoom
        self.color = int(self.radar_config['color'])  # 0-8
        self.smooth = int(self.radar_config['smooth'])  # blur (1) or not (0) radar data.
        self.snow = int(self.radar_config['snow'])  # display (1) or not (0) snow

    def get_image(self):
        """
        Download observation image of rain map.
        """
        # zoom compensation for radar image size 512x512 px.
        zoom = self.zoom
        if self.size == 512:
            zoom = self.zoom - 1
        url = (f'https://tilecache.rainviewer.com/v2/radar{self.path}/{self.size}/{zoom}/'
               f'{self.latitude}/{self.longitude}/{self.color}/{self.smooth}_{self.snow}.png')
        # Get image from url of Rain viewer API
        loading = True
        while loading:
            try:
                with urllib.request.urlopen(url) as url_image:
                    image = np.asarray(bytearray(url_image.read()), dtype="uint8")
            except (urllib.error.URLError, urllib.error.HTTPError) as error:
                print(error)
                print("Rain img download failed, trying again...")
                time.sleep(10)  # Wait before retry.
            finally:
                self.img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                loading = False

    def get_rain_status(self):
        """
        Process downloaded image.
        Draw annotated image with frames and rain info.
        Set rain status for this observation.
        """
        img = np.copy(self.img)
        self.img_annotation = np.zeros_like(img)
        img_origo = cut_low_rain(img.copy(), int(self.radar_config['rain_threshold']), 255)
        status = VALUES.status_ok
        full = float(img.shape[0])
        half = full / 2.0
        mercator = GlobalMercator(tileSize=256)
        meters_x, meters_y = mercator.LatLonToMeters(self.latitude, self.longitude)
        pixels_x = mercator.MetersToPixels(meters_x, meters_y, self.zoom)[0]  # only x

        # Calculate the longitude and latitude coordinates of a point 1 kilometer away origin
        distance = 1  # 1 km
        lon2, lat2 = get_destination_point(self.longitude, self.latitude, 90, distance)
        meters_x1, meters_y1 = mercator.LatLonToMeters(lat2, lon2)

        # Convert coord of lat/lon in meters to pixels coords x and calculate pixels to 1 km
        pixels_x1 = mercator.MetersToPixels(meters_x1, meters_y1, self.zoom)[0]  # only x
        km_to_px = pixels_x1 - pixels_x

        # Detect rain in each frame
        frame_list = [{'distance': int(self.radar_config['frame_warning']),
                       'frame': VALUES.status_warn},
                      {'distance': int(self.radar_config['frame_alert']),
                       'frame': VALUES.status_alert},
                      {'distance': int(self.radar_config['frame_rain']),
                       'frame': VALUES.status_rain}]
        percentage = 0
        img_list = [cut_low_rain(img, int(self.radar_config['rain_threshold']), 255),
                    self.img_annotation]
        for frame in frame_list:

            # Set frame size for RAIN detection
            dist = float(frame['distance'])
            top = int(half - (dist * km_to_px))
            bottom = int(half + (dist * km_to_px))
            left = int(half - (dist * km_to_px))
            right = int(half + (dist * km_to_px))

            # Detect rain and get results
            percentage, status = detect_rain(img_origo, top, bottom, left, right, frame, status)
            if frame['frame'] == VALUES.status_warn:
                self.status_percentage_warn = percentage
            if frame['frame'] == VALUES.status_alert:
                self.status_percentage_alert = percentage
            if frame['frame'] == VALUES.status_rain:
                self.status_percentage_rain = percentage

            # Draw image frame with annotation
            for img_f in img_list:
                img_f = cv2.rectangle(img_f,
                                      (top, left),
                                      (bottom, right),
                                      VALUES.color_deepskyblue,
                                      thickness=1)
                cv2.putText(img_f,
                            f"{frame['frame']} {percentage:.1f}%",
                            (left + 2, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            VALUES.color_purple,
                            1)

        # Image main frame and annotations
        color_status = VALUES.color_green
        if status == VALUES.status_rain:
            color_status = VALUES.color_red
        if status == VALUES.status_alert:
            color_status = VALUES.color_red
        if status == VALUES.status_warn:
            color_status = VALUES.color_yellow
        for img_f in img_list:
            # Frame around
            img_f = cv2.rectangle(img_f, (1, 1), (int(full), int(full)), color_status, thickness=2)

            # Date and status on bottom left
            img_f = cv2.rectangle(img_f,
                                  (3, int(full) - 16),
                                  (int(full) - 2, int(full) - 2),
                                  VALUES.color_dark_grey,
                                  -1)
            cv2.putText(img_f,
                        '{} {}'.format(self.date_str, status),
                        (3, int(full) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color_status,
                        1)

            # Write type of observation on bottom right corner
            text_size = cv2.getTextSize('{}'.format(self.type),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        1)
            cv2.putText(img_f,
                        '{}'.format(self.type),
                        (int(full) - (int(text_size[0][0] + 5)),
                         int(full) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        VALUES.color_white,
                        1)

        self.img_annotated = img_list[0]
        self.img_annotation = img_list[1]
        self.status = status
        self.status_percentage = percentage

    def show_image(self):
        """
        Show annotated image in window. Used for debug.
        """
        cv2.imshow("Rain status {} at {}".format(self.status, self.date_str), self.img_annotated)
        cv2.waitKey(0)

    def get_img_with_map(self) -> np.ndarray:
        """
        Assemble the final image with map as background.

        :return: np_array img contain map, rain and frame layers
        """
        img_map_tail = self.map_tile
        img_in = self.img
        img_in_frame = self.img_annotation
        img_in_frame_alpha = add_alpha_channel(img_in_frame, 215, 255)
        img_cut_rain = cut_low_rain(img_in, int(self.radar_config['rain_threshold']), 255)
        img_blur_rain = blur_rain(img_cut_rain)
        img_alpha_ch = add_alpha_channel(img_blur_rain, 215, 255)
        img_blend_map = blend_map_rain(img_alpha_ch, img_map_tail)
        img_blend_map_light = blend_map_rain_light(img_blend_map, img_blur_rain, img_map_tail)
        img_framed = draw_frame(img_blend_map_light, img_in_frame_alpha)
        return img_framed
