"""
Module used to get map tile with target point in the center.
"""
import time
import urllib.request
import urllib.error

import numpy as np
import cv2

from weather_alert.globalmaptiles import GlobalMercator


class MapTile:
    """
    Class used to get map tile with target point in the center.
    """

    def __init__(self, lat: float, lon: float, zoom: int, size: int, use_map: int):
        """
        :param lat: Latitude of target point
        :param lon: Longitude of target point
        :param zoom: Map zoom level
        :param size: Size of the map tile
        :param use_map: Which map to use (0 - no map, 1 - OSM, 2 - GM satellite)
        """
        self.lat = lat
        self.lon = lon
        self.size = size
        self.zoom = zoom
        self.use_map = use_map
        self.dist_t = 2
        self.mercator = GlobalMercator(256)  # 256 is tile size always
        self.meters_x, self.meters_y = self.mercator.LatLonToMeters(self.lat, self.lon)
        self.tile_x, self.tile_y = self.mercator.MetersToTile(self.meters_x,
                                                              self.meters_y,
                                                              self.zoom)
        self.tminx, self.tminy = [self.tile_x - self.dist_t, self.tile_y - self.dist_t]
        self.tmaxx, self.tmaxy = [self.tile_x + self.dist_t, self.tile_y + self.dist_t]

    def get_map_with_neighbours(self) -> np.ndarray:
        """
        Get target tile with tiles around as map.

        :return: np.ndarray: image of the map
        """
        imgv_list = []
        url = ""
        for tile_y in range(self.tminy, self.tmaxy + 1):
            imgh_list = []
            for tile_x in range(self.tminx, self.tmaxx + 1):
                google_x, google_y = self.mercator.GoogleTile(tile_x, tile_y, self.zoom)
                if self.use_map == 1:
                    url = (f'http://localhost:8082/tiles/1.0.0/osm_demo/webmercator/'
                           f'{self.zoom}/{google_x}/{google_y}.png')
                if self.use_map == 2:
                    url = (f'http://localhost:8082/tiles/1.0.0/gm_layer/gm_grid/'
                           f'{self.zoom}/{google_x}/{google_y}.png')
                loading = True
                while loading:
                    try:
                        with urllib.request.urlopen(url) as url_image:
                            image = np.asarray(bytearray(url_image.read()), dtype="uint8")
                            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                            imgh_list.append(img)
                            loading = False
                    except (urllib.error.URLError, urllib.error.HTTPError) as error:
                        print("Map tile download failed, trying again...")
                        print(error)
                        time.sleep(10)  # Wait before retry.
            imgh = cv2.hconcat(imgh_list)  # Concatenate images horizontally
            imgv_list.append(imgh)
        imgv_list.reverse()
        img = cv2.vconcat(imgv_list)  # Concatenate images vertically
        return img

    def get_tile(self) -> np.ndarray:
        """
        # Get target tile with tiles around and create map tile of
        specified size with target point in the center or empty img.

        :return: image of specified size
        """
        if self.use_map == 0:
            # Create a blank black image
            image = np.zeros((self.size, self.size, 3), np.uint8)
            return image

        # Get map with target point in the center
        img = self.get_map_with_neighbours()

        # Get begin of map tail raster in meters
        bounds_max = self.mercator.TileBounds(self.tminx, self.tmaxy, self.zoom)
        bounds_min = self.mercator.TileBounds(self.tmaxx, self.tminy, self.zoom)
        begin_x = bounds_max[0]
        begin_y = bounds_min[1]

        # Convert coord of lat/lon in meters to pixels coords
        pixel_x, pixel_y = self.mercator.MetersToPixels(self.meters_x, self.meters_y, self.zoom)
        begin_px_x, begin_px_y = self.mercator.MetersToPixels(begin_x, begin_y, self.zoom)

        # Get coord of lat/lon in pixels in image xy coords
        img_px_x = int(pixel_x - begin_px_x)
        img_px_y = int(pixel_y - begin_px_y)
        full = int(img.shape[0])
        img_px_y = full - img_px_y

        # Draw circle on lat/lon in image
        img = cv2.circle(img, (img_px_x, img_px_y), radius=4, color=(0, 0, 255), thickness=-1)

        # Get map tile of specified size with target point in the center
        right_x = img_px_x + int(self.size/2)
        left_x = img_px_x - int(self.size/2)
        top_y = img_px_y - int(self.size/2)
        bottom_y = img_px_y + int(self.size/2)
        img_crop = img[top_y:bottom_y, left_x:right_x]

        return img_crop
