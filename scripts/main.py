"""
Used to run the program. It reads the config file, gets the data from the RainViewer API,
processes it, and shows the images and data. It then enters a loop where it checks for new
data every 20 seconds. It's helpful for tune the configuration to your needs.
"""
import os
import time
import yaml

from weather_alert.rainviewer import RainViewer
from weather_alert.rainviewer import show


def main():
    """
    Load config file, get data from RainViewer API, process it, and show the images and data.
    :return: None
    """
    conf_dir = os.path.dirname(os.path.realpath(__file__))
    conf_dir = "{}/../config/config.yaml".format(conf_dir)
    config = load_params_from_yaml(conf_dir)
    default_location = config['general']['default_location']
    lat = float(config[default_location]['lat'])
    lon = float(config[default_location]['lon'])
    radar_config = dict(config['radar'])
    use_map = int(config['general']['use_map'])

    print('Getting data from Rain viewer API...')
    rain_viewer = RainViewer(lat, lon, radar_config, use_map)
    rain_viewer.process_data()
    rain_viewer.print_rain_status()
    show(rain_viewer.observations[-1])
    rain_viewer.show_now_and_forecast()
    rain_viewer.show_images()

    check_interval = 20
    while True:
        # Checking for fresh map list...
        result = rain_viewer.update_data()
        if result:
            print('New map list received.')
            rain_viewer.process_data()
            rain_viewer.print_rain_status()
            show(rain_viewer.observations[-1])
            rain_viewer.show_now_and_forecast()
            rain_viewer.show_images()
        time.sleep(check_interval)


def load_params_from_yaml(config_path: str) -> dict:
    """
    Load parameters from the config file.

    :param config_path: as string
    :return: config data as a dictionary
    """
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data


if __name__ == "__main__":
    main()
