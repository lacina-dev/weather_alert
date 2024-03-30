"""
Helper file with functions to calculate distance between two points
and destination point given a starting point in WGS84 coordinates.
"""
import math


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate circle distance between two points
    on the earth in decimal degrees.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine computing
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    hav_ang = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    central_angle = 2 * math.atan2(math.sqrt(hav_ang), math.sqrt(1 - hav_ang))
    distance = 6371 * central_angle  # 6371 Radius of Earth in kilometers

    return distance


def get_destination_point(lon: float, lat: float, bearing: int, distance: float) -> [float, float]:
    """
    Calculate the destination point given a starting point, bearing, and distance
    """
    # Convert decimal degrees to radians
    lon, lat = map(math.radians, [lon, lat])

    # Radius of the Earth in km
    earth_r = 6371.0

    # Convert distance from km to rad
    distance = distance / earth_r

    # Convert bearing to radians
    bearing = math.radians(bearing)

    # Calculate destination point
    lat2 = math.asin(math.sin(lat) * math.cos(distance) +
                     math.cos(lat) * math.sin(distance) * math.cos(bearing))
    lon2 = lon + math.atan2(math.sin(bearing) * math.sin(distance) * math.cos(lat),
                            math.cos(distance) - math.sin(lat) * math.sin(lat2))

    # Convert radians to decimal degrees
    lon2 = math.degrees(lon2)
    lat2 = math.degrees(lat2)

    return lon2, lat2
