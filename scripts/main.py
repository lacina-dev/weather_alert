"""
Used to run the program. It reads the config file, gets the data from the configured
radar source (RainViewer or ČHMÚ), processes it, and shows the images and data.
It then enters a loop where it checks for new data every 20 seconds.
It's helpful for tuning the configuration to your needs.
"""
import os
import time
import yaml

from weather_alert.rainviewer import RainViewer
from weather_alert.rainviewer import show
from weather_alert.chmi import Chmi


def _build_source(source: str, lat: float, lon: float,
                  radar_config: dict, use_map: int):
    """Instantiate and return the configured radar source object."""
    if source == "chmi":
        return Chmi(lat, lon, radar_config, use_map)
    return RainViewer(lat, lon, radar_config, use_map)


def main():
    """
    Load config file, get data from the selected radar source,
    process it, and show the images and data.
    :return: None
    """
    conf_dir = os.path.dirname(os.path.realpath(__file__))
    conf_dir = f"{conf_dir}/../config/config.yaml"
    config = load_params_from_yaml(conf_dir)
    default_location = config['general']['default_location']
    lat = float(config[default_location]['lat'])
    lon = float(config[default_location]['lon'])
    radar_config = dict(config['radar'])
    use_map = int(config['general']['use_map'])
    source = str(config['general'].get('source', 'rainviewer')).lower()

    print(f'Getting data from {source.upper()} radar source...')
    radar = _build_source(source, lat, lon, radar_config, use_map)
    radar.process_data()
    radar.print_rain_status()

    if source == 'rainviewer':
        show(radar.observations[-1])
    radar.show_now_and_forecast()
    radar.show_images()

    check_interval = 20
    while True:
        result = radar.update_data()
        if result:
            print('New map list received.')
            radar.process_data()
            radar.print_rain_status()
            if source == 'rainviewer':
                show(radar.observations[-1])
            radar.show_now_and_forecast()
            radar.show_images()
        time.sleep(check_interval)


def load_params_from_yaml(config_path: str) -> dict:
    """
    Load parameters from the config file.

    :param config_path: as string
    :return: config data as a dictionary
    """
    with open(config_path, 'r', encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return config_data


if __name__ == "__main__":
    main()
