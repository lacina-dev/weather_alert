"""
ČHMÚ (Czech Hydrometeorological Institute) radar data source.

Provides current radar composites and 60-minute nowcast from:
https://opendata.chmi.cz/meteorology/weather/radar/composite/

Data details (from HDF5 metadata):
  Product:    CAPPI at 2 km altitude (pseudocappi2km)
  Projection: Mercator (same as Google Maps / EPSG:3857)
  Image size: 680 × 460 px
  Coverage:   LL lon=11.2669, lat=48.0473  /  UR lon=19.6240, lat=51.4584
  Resolution: ~1555 m/pixel (~0.88 km/px at 50 °N)
  Update:     every 5 minutes
  Nowcast:    6 steps × 10 min = 60 min ahead (one tar per run)

Color palette (BGR):
  [  0,   0,   0] – no echo / outside coverage
  [196, 196, 196] – very light echo (below rain_threshold → ignored)
  [112,   0,  56] – light rain  ~10 dBZ
  [168,   0,  48] – moderate rain ~30 dBZ
  [192, 108,   0] – heavy rain  ~45 dBZ
  [252,   0,   0] – very heavy  ~55 dBZ
"""

import io
import math
import re
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from weather_alert.coord_helper import get_destination_point
from weather_alert.map_tile import MapTile
from weather_alert.values import Values

VALUES = Values()

# ── Geographic bounds from ČHMÚ HDF5 metadata ──────────────────────────────
# IMPORTANT: the published PNG (680×460) carries a black border around the
# real radar grid. The actual data subarea is taken from the HDF5
# metadata (xsize/ysize) and starts at (DATA_X0, DATA_Y0) inside the PNG.
# Bounds (LL/UR lat/lon) describe the corners of the *data* area only, so
# pixel maths must offset by the border or positions are wrong by ~64 km E
# and ~127 km N.
_LL_LON = 11.266869
_LL_LAT = 48.047275
_UR_LON = 19.623974
_UR_LAT = 51.458369
_IMG_W   = 680   # full PNG width  [px]
_IMG_H   = 460   # full PNG height [px]
_DATA_X0 = 1     # left edge of radar data inside PNG  [px]
_DATA_Y0 = 82    # top  edge of radar data inside PNG  [px]
_DATA_W  = 598   # radar data width  [px]  (HDF5 xsize)
_DATA_H  = 377   # radar data height [px]  (HDF5 ysize, allowing 1-px slack)

# How many of the most recent CAPPI frames we keep (and download). The web
# UI shows max 60 min of "past" + NOW, so 13 is plenty (5-min cadence).
_KEEP_OBS = 13

# Base URLs
_BASE = "https://opendata.chmi.cz/meteorology/weather/radar/composite"
_CURRENT_URL  = f"{_BASE}/pseudocappi2km/png/"
_NOWCAST_URL  = f"{_BASE}/fct_pseudocappi2km/png/"


def _merc_y(lat_deg: float) -> float:
    """Mercator Y coordinate for a latitude in degrees."""
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))


def _latlon_to_pixel_f(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert WGS84 lat/lon to FLOATING-POINT pixel coordinates in the
    ČHMÚ composite PNG. Returns (col, row) – row 0 is top (north).
    The bounds describe the *data* sub-area, not the whole PNG, so we
    add the (DATA_X0, DATA_Y0) border offset.
    """
    col = _DATA_X0 + (lon - _LL_LON) / (_UR_LON - _LL_LON) * _DATA_W
    y_ll = _merc_y(_LL_LAT)
    y_ur = _merc_y(_UR_LAT)
    lat_frac = (_merc_y(lat) - y_ll) / (y_ur - y_ll)
    row = _DATA_Y0 + (1.0 - lat_frac) * _DATA_H
    return col, row


def _latlon_to_pixel(lat: float, lon: float) -> Tuple[int, int]:
    """Integer pixel coordinates (kept for callers that index arrays)."""
    col, row = _latlon_to_pixel_f(lat, lon)
    return int(round(col)), int(round(row))


def _km_to_pixels(lat: float, lon: float, distance_km: float) -> float:
    """
    Return the FRACTIONAL pixel distance corresponding to *distance_km*
    at the given position using the same east-bearing approach as in
    observation.py.  Must be float — ČHMÚ scale is ~0.88 km/px so
    rounding 1 km to an integer would silently break the geometry.
    """
    lon2, lat2 = get_destination_point(lon, lat, 90, distance_km)
    col0, _ = _latlon_to_pixel_f(lat, lon)
    col1, _ = _latlon_to_pixel_f(lat2, lon2)
    return max(1e-6, abs(col1 - col0))


# ── ČHMÚ palette helpers ───────────────────────────────────────────────────
# Pixel values that are NOT rain. Anything else in the image is treated as a
# real radar return.
_NON_RAIN_BGR = (
    (0, 0, 0),         # outside coverage / no echo
    (196, 196, 196),   # very-light "below noise floor" echo → ignored
)


def _rain_mask(img: np.ndarray) -> np.ndarray:
    """
    Return a uint8 mask (255 where pixel represents rain, 0 otherwise).
    Works directly on the ČHMÚ palette without relying on brightness.
    """
    h, w = img.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for bgr in _NON_RAIN_BGR:
        match = cv2.inRange(img, np.array(bgr, dtype=np.uint8),
                            np.array(bgr, dtype=np.uint8))
        mask[match > 0] = 0
    return mask


def _list_remote(url: str) -> List[str]:
    """Return href filenames found in a directory listing at *url*."""
    req = urllib.request.Request(
        url + ('&' if '?' in url else '?') + f'_={int(time.time())}',
        headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'},
    )
    with urllib.request.urlopen(req) as r:
        content = r.read().decode()
    return re.findall(r'href="([^"]+)"', content)


def _download_bytes(url: str, retries: int = 3, delay: int = 10) -> bytes:
    """Download URL with retry on error."""
    req = urllib.request.Request(
        url,
        headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"ČHMÚ download error ({url}): {exc}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _parse_timestamp_from_name(filename: str) -> int:
    """
    Extract UTC Unix timestamp from ČHMÚ filename.
    Patterns:
      pacz2gmaps3.z_cappi020.20260406.1135.0.png       → 20260406 1135
      pacz2gmaps3.fct_z_cappi020.20260406.1145.10.png  → 20260406 1145
    """
    m = re.search(r'\.(\d{8})\.(\d{4})\.', filename)
    if not m:
        return 0
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    return int(dt.timestamp())


# ── ChmiObservation ─────────────────────────────────────────────────────────

class ChmiObservation:
    """
    One ČHMÚ radar frame.  Mirrors the public interface of Observation so
    that the rest of the codebase can treat both interchangeably.
    """

    def __init__(self,
                 png_data: bytes,
                 filename: str,
                 timestamp: int,
                 lat: float,
                 lon: float,
                 radar_config: dict,
                 observation_type: str,
                 map_tile: np.ndarray,
                 half_px: int = 0,
                 half_km: float = 0.0):

        self.time = timestamp
        self.date_str = datetime.fromtimestamp(self.time)
        self.type = observation_type
        self.latitude = lat
        self.longitude = lon
        self.radar_config = radar_config
        self.map_tile = map_tile
        self.status = VALUES.status_ok
        self.status_percentage = 0.0
        self.status_percentage_warn = 0.0
        self.status_percentage_alert = 0.0
        self.status_percentage_rain = 0.0
        self.img = None
        self.img_annotated = None
        self.img_annotation = None
        self._png_data = png_data

        self._size = int(radar_config.get('size', 512))
        # Crop geometry shared with Chmi (which built the OSM map at the
        # exact same geographic extent). When not provided (e.g. tests)
        # falls back to the same formula get_image() used to use.
        if half_px > 0 and half_km > 0:
            self._half_px = int(half_px)
            self._half_km = float(half_km)
        else:
            km_px = _km_to_pixels(self.latitude, self.longitude, 1.0)
            fw = float(radar_config.get('frame_warning', 20))
            self._half_px = max(int(fw * km_px * 1.2) + 2, 32)
            self._half_km = self._half_px / float(km_px)

    # ── image loading ────────────────────────────────────────────────────────

    def get_image(self):
        """
        Decode the raw PNG bytes and crop a square region centred on the
        robot position.  The crop is resized to *size* × *size* pixels so
        that downstream analysis (detect_rain / annotated image) is
        identical to the RainViewer path.
        """
        full = cv2.imdecode(np.frombuffer(self._png_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR)
        if full is None:
            raise RuntimeError("ČHMÚ: failed to decode radar image")

        col, row = _latlon_to_pixel(self.latitude, self.longitude)

        # Crop half-extent in source-image pixels was pre-computed by Chmi
        # so the OSM map background and this crop share scale exactly.
        half = self._half_px

        # Clamp to image bounds
        r0, r1 = max(0, row - half), min(_IMG_H, row + half)
        c0, c1 = max(0, col - half), min(_IMG_W, col + half)

        crop = full[r0:r1, c0:c1]

        # Translate robot position into crop-local coordinates and pad
        # so the robot is exactly at the centre of a square canvas.
        local_row = row - r0
        local_col = col - c0
        crop_h, crop_w = crop.shape[:2]

        # Build square canvas (robot centre)
        sq = max(crop_h, crop_w, 2 * half)
        canvas = np.zeros((sq, sq, 3), dtype=np.uint8)
        off_r = sq // 2 - local_row
        off_c = sq // 2 - local_col
        dr0 = max(0, off_r)
        dc0 = max(0, off_c)
        sr0 = max(0, -off_r)
        sc0 = max(0, -off_c)
        paste_h = min(crop_h - sr0, sq - dr0)
        paste_w = min(crop_w - sc0, sq - dc0)
        if paste_h > 0 and paste_w > 0:
            canvas[dr0:dr0 + paste_h, dc0:dc0 + paste_w] = \
                crop[sr0:sr0 + paste_h, sc0:sc0 + paste_w]

        # ČHMÚ grey „no-data" background → make black so detect_rain works
        grey_mask = np.all(canvas == [196, 196, 196], axis=2)
        canvas[grey_mask] = [0, 0, 0]

        self.img = cv2.resize(canvas, (self._size, self._size),
                              interpolation=cv2.INTER_NEAREST)

    # ── rain analysis ────────────────────────────────────────────────────────

    def get_rain_status(self):
        """
        Analyse the cropped image and produce annotated output.
        Logic mirrors observation.py: cut_low_rain → detect_rain per zone.
        """
        img = np.copy(self.img)
        self.img_annotation = np.zeros_like(img)
        rain_thresh = int(self.radar_config.get('rain_threshold', 30))
        img_origo = _cut_low_rain_chmi(img.copy(), rain_thresh)
        status = VALUES.status_ok
        full = float(self._size)
        half = full / 2.0

        # px-per-km in the resized crop. self._half_km is the geographic
        # half-extent of the resized image, shared with the OSM map tile
        # so frame rectangles align perfectly with map features.
        km_to_px = (self._size / 2.0) / float(self._half_km)

        frame_list = [
            {'distance': float(self.radar_config.get('frame_warning', 20)),
             'frame': VALUES.status_warn},
            {'distance': float(self.radar_config.get('frame_alert', 9)),
             'frame': VALUES.status_alert},
            {'distance': float(self.radar_config.get('frame_rain', 1)),
             'frame': VALUES.status_rain},
        ]
        percentage = 0.0
        img_list = [img_origo, self.img_annotation]

        for frame in frame_list:
            dist = frame['distance']
            top    = int(half - dist * km_to_px)
            bottom = int(half + dist * km_to_px)
            left   = int(half - dist * km_to_px)
            right  = int(half + dist * km_to_px)

            percentage, status = _detect_rain_chmi(img_origo, top, bottom,
                                                   left, right, frame, status)
            if frame['frame'] == VALUES.status_warn:
                self.status_percentage_warn = percentage
            if frame['frame'] == VALUES.status_alert:
                self.status_percentage_alert = percentage
            if frame['frame'] == VALUES.status_rain:
                self.status_percentage_rain = percentage

            for img_f in img_list:
                cv2.rectangle(img_f, (left, top), (right, bottom),
                              VALUES.color_deepskyblue, thickness=1)
                cv2.putText(img_f,
                            f"{frame['frame']} {percentage:.1f}%",
                            (left + 2, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            VALUES.color_purple, 1)

        color_status = VALUES.color_green
        if status in (VALUES.status_rain, VALUES.status_alert):
            color_status = VALUES.color_red
        elif status == VALUES.status_warn:
            color_status = VALUES.color_yellow

        for img_f in img_list:
            cv2.rectangle(img_f, (1, 1), (int(full), int(full)),
                          color_status, thickness=2)
            cv2.rectangle(img_f,
                          (3, int(full) - 16),
                          (int(full) - 2, int(full) - 2),
                          VALUES.color_dark_grey, -1)
            cv2.putText(img_f,
                        f"{self.date_str} {status}",
                        (3, int(full) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_status, 1)
            text_size = cv2.getTextSize(self.type, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img_f, self.type,
                        (int(full) - (int(text_size[0][0]) + 5), int(full) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, VALUES.color_white, 1)

        self.img_annotated = img_list[0]
        self.img_annotation = img_list[1]
        self.status = status
        self.status_percentage = percentage

    # ── display helpers ──────────────────────────────────────────────────────

    def get_img_with_map(self) -> np.ndarray:
        """
        Return the radar frame blended on top of the OSM background tile so
        that the "Rain now" view in the web UI looks similar to the legacy
        RainViewer output. Falls back to the bare annotated image when no
        map background is available.
        """
        if self.img_annotated is None:
            return self.img if self.img is not None else np.zeros((1, 1, 3),
                                                                  dtype=np.uint8)

        if self.map_tile is None or self.map_tile.size == 0:
            return self.img_annotated

        # ``map_tile`` is RGBA from MapTile via cv2.cvtColor(...RGB2RGBA);
        # collapse it to BGR for compositing with the radar/annotation BGR
        # images. When use_map=0 the tile is all-black so the result equals
        # img_annotated.
        if self.map_tile.ndim == 3 and self.map_tile.shape[2] == 4:
            map_bgr = cv2.cvtColor(self.map_tile, cv2.COLOR_RGBA2BGR)
        else:
            map_bgr = self.map_tile

        if map_bgr.shape[:2] != self.img_annotated.shape[:2]:
            map_bgr = cv2.resize(map_bgr,
                                 (self.img_annotated.shape[1],
                                  self.img_annotated.shape[0]),
                                 interpolation=cv2.INTER_AREA)

        # Build a colour-based mask of rain pixels in the radar image.
        radar = self.img if self.img is not None else self.img_annotated
        rain_mask = _rain_mask(radar)
        rain_mask_inv = cv2.bitwise_not(rain_mask)

        # Slightly darken the map for contrast.
        map_dark = cv2.addWeighted(map_bgr, 0.85,
                                   np.zeros_like(map_bgr), 0.15, 0)
        bg = cv2.bitwise_and(map_dark, map_dark, mask=rain_mask_inv)
        fg = cv2.bitwise_and(radar, radar, mask=rain_mask)
        blended = cv2.add(bg, fg)

        # Re-apply the annotation overlay (frames + status text) on top.
        if self.img_annotation is not None:
            ann_gray = cv2.cvtColor(self.img_annotation, cv2.COLOR_BGR2GRAY)
            _, ann_mask = cv2.threshold(ann_gray, 1, 255, cv2.THRESH_BINARY)
            ann_mask_inv = cv2.bitwise_not(ann_mask)
            blended = cv2.add(
                cv2.bitwise_and(blended, blended, mask=ann_mask_inv),
                cv2.bitwise_and(self.img_annotation,
                                self.img_annotation, mask=ann_mask),
            )
        return blended

    def show_image(self):
        cv2.imshow(f"ČHMÚ {self.status} at {self.date_str}", self.img_annotated)
        cv2.waitKey(0)


# ── helper image-processing functions ───────────────────────────────────────

def _cut_low_rain_chmi(img: np.ndarray, thresh: int) -> np.ndarray:
    """
    Keep only pixels that represent actual rain echo. ČHMÚ uses a fixed
    palette so we mask by colour rather than brightness — the brightness
    threshold previously used dropped most rain shades (gray range 28–196).
    The *thresh* argument is kept for API compatibility but unused.
    """
    del thresh  # legacy parameter, ignored
    mask = _rain_mask(img)
    return cv2.bitwise_and(img, img, mask=mask)


def _detect_rain_chmi(img_origo: np.ndarray, top: int, bottom: int, left: int,
                      right: int, frame: dict, status: str) -> Tuple[float, str]:
    """Count rain pixels in zone as percentage of zone area."""
    img_crop = img_origo[top:bottom, left:right]
    if img_crop.size == 0:
        return 0.0, status
    mask = _rain_mask(img_crop)
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0, status
    rain_px = int(cv2.countNonZero(mask))
    percentage = float(np.round(100.0 * rain_px / total, 2))
    if percentage > 0:
        status = frame['frame']
    return percentage, status


# ── Chmi (main class, mirrors RainViewer) ───────────────────────────────────

class Chmi:
    """
    Fetches current radar + 60-min nowcast from ČHMÚ Open Data.
    Exposes the same public interface as RainViewer so main.py can swap
    the two sources transparently.
    """

    def __init__(self, lat: float, lon: float, radar_config: dict, use_map: int):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.radar_config = radar_config
        self.use_map = use_map
        self.observations: List[ChmiObservation] = []
        self.nowcasts: List[ChmiObservation] = []
        self.generated = 0
        # Compute the radar crop geometry ONCE so the OSM map background
        # and every observation crop share the exact same geographic
        # extent (otherwise the rain would not align with map features).
        frame_warning_km = float(self.radar_config.get('frame_warning', 20))
        km_px = _km_to_pixels(self.latitude, self.longitude, 1.0)
        self._half_px = max(int(frame_warning_km * km_px * 1.2) + 2, 32)
        self._half_km = self._half_px / float(km_px)
        self.map_tile = self._build_map_tile()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _build_map_tile(self) -> np.ndarray:
        """
        Build an OpenStreetMap background whose geographic extent matches
        the radar crop pixel-for-pixel. The radar crop covers a square of
        side ``2 * self._half_km`` km in ``size`` px (see
        ``ChmiObservation``); we pick the largest OSM zoom whose native
        m/px is still ≥ the required m/px, fetch a slightly larger tile,
        then resize so 1 km on the map = 1 km on the radar.
        """
        size = int(self.radar_config.get('size', 512))
        side_m = 2.0 * self._half_km * 1000.0
        required_m_per_px = side_m / float(size)

        lat_rad = math.radians(self.latitude)
        # OSM Web Mercator m/px at given lat: 156543.03 * cos(lat) / 2^Z
        base = 156543.03 * max(math.cos(lat_rad), 1e-6)
        # Largest Z with osm_m_per_px >= required (smaller m/px = more zoom).
        zoom = int(math.floor(math.log2(base / required_m_per_px)))
        zoom = max(1, min(zoom, 19))
        osm_m_per_px = base / (2 ** zoom)

        # Pixels of OSM imagery that span the same geographic extent as the
        # radar crop. Round up so we never undershoot the target area.
        size_osm = int(math.ceil(side_m / osm_m_per_px))
        size_osm = max(size_osm, 16)

        img = MapTile(self.latitude, self.longitude, zoom, size_osm,
                      self.use_map).get_tile()
        if img is None or img.size == 0:
            img = np.zeros((size, size, 3), dtype=np.uint8)
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    def _make_observation(self, png_data: bytes, filename: str,
                          obs_type: str) -> ChmiObservation:
        ts = _parse_timestamp_from_name(filename)
        obs = ChmiObservation(png_data, filename, ts,
                              self.latitude, self.longitude,
                              self.radar_config, obs_type, self.map_tile,
                              half_px=self._half_px,
                              half_km=self._half_km)
        obs.get_image()
        obs.get_rain_status()
        return obs

    # ── public API ───────────────────────────────────────────────────────────

    def get_map_data(self):
        """
        Fetch the latest current radar PNG files (past ~60 min) and the
        most-recent nowcast tar (6 × 10-min steps, up to +60 min).
        """
        # ── current observations ──────────────────────────────────────────
        try:
            names = _list_remote(_CURRENT_URL)
        except Exception as exc:
            print(f"ČHMÚ: cannot list current radar: {exc}")
            return

        png_names = sorted(
            n for n in names if re.match(r'pacz2gmaps3\.z_cappi020\.\d{8}\.\d{4}\.0\.png', n)
        )
        if not png_names:
            print("ČHMÚ: no current radar files found")
            return

        # ČHMÚ keeps thousands of historical PNGs; we only ever display
        # the last ~13 (60-min ring buffer + NOW). Restrict downloads to
        # the most recent files so startup is fast and recurring polls
        # are cheap.
        png_names = png_names[-_KEEP_OBS:]

        existing_times = {o.time for o in self.observations}
        new_observations = list(self.observations)

        for name in png_names:
            ts = _parse_timestamp_from_name(name)
            if ts in existing_times:
                continue
            try:
                data = _download_bytes(_CURRENT_URL + name)
                obs = self._make_observation(data, name, 'PAST')
                new_observations.append(obs)
                if len(new_observations) > _KEEP_OBS:
                    new_observations.pop(0)
            except Exception as exc:
                print(f"ČHMÚ: skipping {name}: {exc}")

        if new_observations:
            new_observations[-1].type = 'NOW'
            # The annotation strip is baked in by get_rain_status during
            # _make_observation, so re-render the latest frame to refresh
            # the bottom-right "NOW"/"PAST" tag after the type bump.
            new_observations[-1].get_rain_status()
            if len(new_observations) >= 2:
                new_observations[-2].type = 'PAST'
                new_observations[-2].get_rain_status()

        self.observations = new_observations
        if self.observations:
            self.generated = self.observations[-1].time

        # ── nowcast ───────────────────────────────────────────────────────
        try:
            tar_names = _list_remote(_NOWCAST_URL)
        except Exception as exc:
            print(f"ČHMÚ: cannot list nowcast: {exc}")
            return

        tar_names = sorted(
            n for n in tar_names if n.endswith('.tar')
        )
        if not tar_names:
            return

        latest_tar = tar_names[-1]
        try:
            tar_data = _download_bytes(_NOWCAST_URL + latest_tar)
        except Exception as exc:
            print(f"ČHMÚ: cannot download nowcast tar: {exc}")
            return

        nowcasts: List[ChmiObservation] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_data)) as tf:
                members = sorted(m for m in tf.getnames() if m.endswith('.png'))
                for member in members:
                    png_data = tf.extractfile(member).read()
                    basename = member.split('/')[-1]
                    try:
                        obs = self._make_observation(png_data, basename, 'NOWCAST')
                        nowcasts.append(obs)
                    except Exception as exc:
                        print(f"ČHMÚ: skipping nowcast {basename}: {exc}")
        except Exception as exc:
            print(f"ČHMÚ: cannot read nowcast tar: {exc}")
            return

        self.nowcasts = nowcasts

    def process_data(self):
        """Alias kept for interface compatibility with RainViewer."""
        self.get_map_data()

    def update_data(self) -> bool:
        """
        Re-fetch if current data is older than ~5 minutes.

        :return: True if fresh data was fetched.
        """
        if not self.observations:
            self.get_map_data()
            return True
        age = int(time.time()) - self.generated
        if age > 310:
            self.get_map_data()
            return True
        return False

    def get_rain_data(self) -> Dict[str, Union[List[Dict[str, Any]], str, float]]:
        """Return observations summary dict (same structure as RainViewer)."""
        status_list = []
        for obs in self.observations + self.nowcasts:
            status_list.append({
                'time':           obs.time,
                'final_status':   obs.status,
                'date':           str(obs.date_str),
                'type':           obs.type,
                'percent_warn':   obs.status_percentage_warn,
                'percent_alert':  obs.status_percentage_alert,
                'percent_rain':   obs.status_percentage_rain,
            })
        return {
            'generated':      self.generated,
            'observations':   status_list,
            'generated_date': str(datetime.fromtimestamp(self.generated)) if self.generated else '',
            'lat':            self.latitude,
            'lon':            self.longitude,
        }

    def _pick_nowcast(self, offset_min: int,
                      tolerance_s: int = 600) -> Union[ChmiObservation, None]:
        """
        Return the nowcast observation closest to ``NOW + offset_min``.
        Tolerance defaults to one nowcast step (10 min = 600 s).
        """
        if not self.observations or not self.nowcasts:
            return None
        target = self.observations[-1].time + offset_min * 60
        best = min(self.nowcasts, key=lambda o: abs(o.time - target))
        if abs(best.time - target) > tolerance_s:
            return None
        return best

    def _forecast_grid(self) -> np.ndarray:
        """
        Build a 2×2 forecast grid (+10/+20/+30/+40 min) sized identically to
        a single radar frame, so it can sit next to the NOW image at the
        same scale in the web UI.
        """
        size = int(self.radar_config.get('size', 512))
        half = size // 2
        offsets = [10, 20, 30, 40]

        cells = []
        for off in offsets:
            nc = self._pick_nowcast(off)
            if nc is None:
                cell = np.zeros((size, size, 3), dtype=np.uint8)
                cv2.putText(cell, f"+{off} min  N/A",
                            (10, size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            VALUES.color_white, 2)
            else:
                cell = nc.get_img_with_map().copy()
                # Top-left badge with the offset label so users can read
                # the time even after the panel is shrunk.
                label = f"+{off} min"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.7, 2)
                cv2.rectangle(cell, (3, 3), (3 + tw + 8, 3 + th + 8),
                              VALUES.color_dark_grey, -1)
                cv2.putText(cell, label, (7, 3 + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            VALUES.color_white, 2)
            cells.append(cv2.resize(cell, (half, half),
                                    interpolation=cv2.INTER_AREA))

        top = cv2.hconcat([cells[0], cells[1]])
        bot = cv2.hconcat([cells[2], cells[3]])
        return cv2.vconcat([top, bot])

    def evaluate_data(self) -> Tuple[np.ndarray, np.ndarray, bool, bool,
                                     Dict[str, Union[List[Dict[str, Any]], str, float]]]:
        """
        Evaluate all observations.

        :return: (img_map, img_concat, rain_alert, rain_now, rain_data)
        """
        rain_data = self.get_rain_data()
        rain_alert = False
        rain_now = False

        for obs in rain_data['observations'][-4:-1]:
            if (obs['percent_rain']  > self.radar_config.get('min_rain', 5.0) or
                    obs['percent_alert'] > self.radar_config.get('min_alert', 10.0)):
                rain_alert = True
        if (rain_data['observations'] and
                rain_data['observations'][-4]['percent_rain'] >
                self.radar_config.get('min_rain', 5.0)):
            rain_now = True

        # Forecast image is a 2×2 grid of +10/+20/+30/+40 min nowcast
        # frames sized to match a single NOW frame.
        img_concatenated = self._forecast_grid()

        img_map = (self.observations[-1].get_img_with_map()
                   if self.observations else np.zeros((100, 100, 3), np.uint8))
        return img_map, img_concatenated, rain_alert, rain_now, rain_data

    def print_rain_status(self):
        """Print all observation results to stdout."""
        status = self.get_rain_data()
        gen_str = status.get('generated_date', '')
        print(f"ČHMÚ — generated at: {gen_str}")
        for obs in status['observations']:
            print('{:^20} {:^7} {:^7} [{:6.2f}% warn] [{:6.2f}% alert] [{:6.2f}% rain]'
                  .format(obs['date'], obs['final_status'], obs['type'],
                          obs['percent_warn'], obs['percent_alert'], obs['percent_rain']))

    def show_images(self):
        """Show all observation frames in sequence."""
        cv2.namedWindow("ČHMÚ Radar view", cv2.WINDOW_NORMAL)
        for obs in self.observations + self.nowcasts:
            cv2.imshow('ČHMÚ Radar view', obs.get_img_with_map())
            cv2.waitKey(2000)

    def show_now_and_forecast(self):
        """Show current + nowcast strip."""
        if not self.observations:
            return
        frames = [self.observations[-1]] + self.nowcasts
        result = [o.img_annotated for o in frames if o.img_annotated is not None]
        if not result:
            return
        img_concatenated = cv2.hconcat(result)
        cv2.imshow('ČHMÚ Now and nowcasts', img_concatenated)
