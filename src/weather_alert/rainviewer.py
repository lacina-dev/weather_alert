"""
This module managing communication with Rain viewer and proces data of radar observations.
Source of the rain images is https://www.rainviewer.com/.
"""
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Union, Any, Dict

import cv2

from weather_alert.map_tile import MapTile
from weather_alert.observation import Observation
from weather_alert.values import Values

VALUES = Values()


def show(observation: Observation):
    """
    Show observation with map of defined instance.

    :param observation: Observation object to show.
    """
    img = observation.get_img_with_map()
    cv2.imshow('Map', img)


class RainViewer:
    """
    This class managing communication with Rain viewer and proces received images.
    Source of the rain images is https://www.rainviewer.com/
    """
    def __init__(self, lat: float, lon: float, radar_config: dict, use_map: int):
        self.maps_url = "https://api.rainviewer.com/public/weather-maps.json"
        self.json_data = None
        self.version = None
        self.generated = None
        self.host = None
        self.radar_config = radar_config
        self.observations = []
        self.nowcasts = []
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.zoom = int(self.radar_config['zoom'])
        self.use_map = use_map
        self.map_tile = None
        self.get_map_tail()
        self.get_map_data()

    def get_map_tail(self):
        """
        Get map tile from map server or empty img, depend on config.
        """
        print("Loading map from map server...")
        img_map_tail = MapTile(self.latitude,
                               self.longitude,
                               self.zoom,
                               int(self.radar_config['size']),
                               self.use_map).get_tile()
        self.map_tile = cv2.cvtColor(img_map_tail, cv2.COLOR_RGB2RGBA)

    def get_map_data(self):
        """
        Download .json with links to available rain maps.
        """
        loading = True
        while loading:
            try:
                with urllib.request.urlopen(self.maps_url) as url:
                    self.json_data = json.load(url)
            except (urllib.error.URLError, urllib.error.URLError) as error:
                print(error)
                print("Loading failed, trying again...")
                time.sleep(5)
            finally:
                loading = False
        if not loading:

            # Process info data
            self.version = self.json_data['version']
            self.generated = self.json_data['generated']
            self.host = self.json_data['host']


    def timestamp_check(self, timestamp: int) -> bool:
        """
        Check if observation with timestamp is already downloaded.

        :param timestamp:  timestamp to check if already downloaded
        :return: True if observation with timestamp is already downloaded, False otherwise
        """
        result = list(filter(lambda x: x.time == timestamp, self.observations))
        if result:
            return True
        return False

    def process_data(self):
        """
        Process downloaded .json data. Download images and get rain statuses of observations.
        """
        observations = self.observations
        nowcasts = []

        # Radar past (really observed data)
        observation_list_len = len(self.json_data['radar']['past'])
        observation_id = 1
        for observation in self.json_data['radar']['past']:
            observation_type = 'PAST'
            if observation_list_len == observation_id:
                observation_type = 'NOW'
            # Check if radar observation img is already downloaded in the past. If not, save it.
            observation_exist = self.timestamp_check(observation['time'])
            if not observation_exist:
                new_observation = Observation(observation['path'],
                                              observation['time'],
                                              self.latitude,
                                              self.longitude,
                                              self.radar_config,
                                              observation_type,
                                              self.zoom,
                                              self.map_tile)
                new_observation.get_image()
                new_observation.get_rain_status()
                observations.append(new_observation)
                if len(observations) > 13:
                    observations.pop(0)
            observation_id += 1

        # Radar nowcasts (forecasts)
        for observation in self.json_data['radar']['nowcast']:
            observation_type = 'NOWCAST'
            new_observation = Observation(observation['path'],
                                          observation['time'],
                                          self.latitude,
                                          self.longitude,
                                          self.radar_config,
                                          observation_type,
                                          self.zoom,
                                          self.map_tile)
            new_observation.get_image()
            new_observation.get_rain_status()
            nowcasts.append(new_observation)
        self.observations = observations
        self.nowcasts = nowcasts
        self.observations[-2].type = 'PAST'

    def get_rain_data(self) -> Dict[str, Union[List[Dict[str, Any]], str, float]]:
        """
        Get results of all observations.

        :return: Results of all observations
        """
        status_list = list()
        for observation in self.observations + self.nowcasts:
            observation_status = {'time': observation.time,
                                  'final_status': observation.status,
                                  'date': str(datetime.fromtimestamp(observation.time)),
                                  'type': observation.type,
                                  'percent_warn': observation.status_percentage_warn,
                                  'percent_alert': observation.status_percentage_alert,
                                  'percent_rain': observation.status_percentage_rain,
                                  }
            status_list.append(observation_status)
        status = {'generated': self.generated,
                  'observations': status_list,
                  'generated_date': str(datetime.fromtimestamp(self.generated)),
                  'lat': self.latitude,
                  'lon': self.longitude}
        return status

    def evaluate_data(self):
        rain_data = self.get_rain_data()
        print('Evaluate data for last 4 observations:')
        for observation in rain_data['observations'][-5:-1]:
            print('Time: {}  Type: {}  Status: {}  [{:6.2f}% warn] [{:6.2f}% alert] [{:6.2f}% rain]'
                  .format(observation['time'],
                          observation['type'],
                          observation['final_status'],
                          observation['percent_warn'],
                          observation['percent_alert'],
                          observation['percent_rain']))

    def print_rain_status(self):
        """
        Print all radar observations results.
        """
        status = self.get_rain_data()
        print('Generated at: {}  {}'.format(status['generated_date'], status['generated']))
        for observation in status['observations']:
            print('{:^20} {:^7} {:^7} [{:6.2f}% warn] [{:6.2f}% alert] [{:6.2f}% rain]'
                  .format(observation['date'],
                          observation['final_status'],
                          observation['type'],
                          observation['percent_warn'],
                          observation['percent_alert'],
                          observation['percent_rain'],
                          ))


    def update_data(self) -> bool:
        """
        Check if fresh map list is available.

        :return: True if new map list is available, False otherwise
        """
        result = False
        if (int(self.generated) + 310) < int(time.time()):
            self.get_map_data()
            result = True
        return result

    def show_images(self):
        """
        Show all observations as images in one window.
        """
        cv2.namedWindow("Radar view", cv2.WINDOW_NORMAL)
        for observation in self.observations + self.nowcasts:
            img_rad = observation.get_img_with_map()
            cv2.imshow('Radar view', img_rad)
            cv2.waitKey(2000)

    def show_now_and_forecast(self):
        """
        Show now and nowcasts images for next 30 min.
        """
        observations = [self.observations[-1]] + self.nowcasts
        result = [radar.img_annotated for radar in observations]
        img_concatenated = cv2.hconcat(result)
        cv2.imshow('Now and nowcasts', img_concatenated)
