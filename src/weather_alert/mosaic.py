"""
precip-mosaic radar source (``general.source: mosaic``).

Reads the multi-source precipitation mosaic served by the ``vitulus_precip``
HTTP service (default ``http://localhost:8097``) instead of downloading one
radar product directly:

  * ``GET /point``     – time line at the robot: ids of the observed 5-min
                         frames of the last 60 min and of the nowcast frames;
  * ``GET /area.png``  – square cut-out centred on the robot, linear in km
                         (``half_km``/``size`` → ``km_per_px``), meta in the
                         ``X-Precip-Meta`` header:
                           ``style=data``  lossless mm/h + coverage, used for
                                           the frame evaluation;
                           ``style=rate``  unified palette, used for the
                                           pictures (NOW and nowcast frames).
    Nowcast frames are requested as ``frame=nowcast+N`` (minutes after the
    latest observation).

The frame evaluation (rain / alert / warning squares, percentages, statuses)
is the one of the ČHMÚ source: every frame becomes a ``ChmiObservation`` whose
image is black where it is dry and coloured where the rate is at least
``mosaic.rain_mmh``. The published topics therefore keep their format.

Unknown is never reported as dry:
  * service unreachable / error → nothing new is published (the last
    RainAlert gets old, which the mower treats as "no weather data");
  * a frame without radar coverage (or stale data) at the robot has status
    ``UNAVAILABLE``; when that is the NOW frame, no RainAlert is published.
"""
import calendar
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from weather_alert.chmi import Chmi, ChmiObservation
from weather_alert.values import Values

VALUES = Values()

STATUS_UNAVAILABLE = 'UNAVAILABLE'

# update_data asks again this much before poll_s is over (loop jitter)
POLL_SLACK_S = 5.0

DEFAULTS = {
    'url': 'http://localhost:8097',
    'timeout_s': 10.0,       # per HTTP request
    'poll_s': 60,            # how often /point is asked for a new frame
    'rain_mmh': 0.1,         # a pixel at or above this rate is rain
    'half_km': 0.0,          # half side of the cut-out; 0 = 1.6 × frame_warning
    'nowcast_max_min': 60,   # nowcast frames further ahead are not fetched
    'min_covered': 0.5,      # coverage of the rain square needed for a status
}

# The web UI shows 60 min of history + NOW (5-min cadence), as with ČHMÚ.
_KEEP_OBS = 13
_FLAG_STALE = 0x20
# Rain colour (BGR) for rain pixels without a palette colour (rate below the
# palette's lowest bin when rain_mmh < 0.1).
_FALLBACK_BGR = (250, 205, 110)
# Colours the ČHMÚ helpers read as "no rain"; a rain pixel must not use them.
_NO_RAIN_BGR = ((0, 0, 0), (196, 196, 196))


class MosaicError(Exception):
    """precip-mosaic unreachable or answered with an error."""


def parse_frame_id(frame_id: str) -> int:
    """``2026-11-08T09:30Z`` → Unix time (UTC)."""
    return calendar.timegm(time.strptime(frame_id, '%Y-%m-%dT%H:%MZ'))


def decode_data_png(png: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a ``style=data`` image: R,G = round(mm/h·100) u16 big-endian,
    B = type | flags, A = quality (0 = no data).

    :return: (rate in mm/h, NaN where not covered; flag bits of B)
    """
    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise MosaicError('area.png style=data: not an RGBA PNG')
    # OpenCV order is B, G, R, A
    value = (img[:, :, 2].astype(np.uint16) << 8 | img[:, :, 1]).astype(np.float32) / 100.0
    covered = img[:, :, 3] > 0
    rate = np.where(covered, value, np.nan).astype(np.float32)
    flags = (img[:, :, 0] & 0xF8).astype(np.uint8)
    return rate, flags


def decode_colour_png(png: bytes) -> np.ndarray:
    """Decode a ``style=rate`` image (indexed + transparency) to BGRA."""
    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MosaicError('area.png style=rate: not a PNG')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _square(size: int, half_km: float, dist_km: float) -> Tuple[int, int, int, int]:
    """Pixel rectangle (top, bottom, left, right) of a frame square, computed
    exactly like ``ChmiObservation.get_rain_status``."""
    half = size / 2.0
    km_to_px = half / float(half_km)
    return (int(half - dist_km * km_to_px), int(half + dist_km * km_to_px),
            int(half - dist_km * km_to_px), int(half + dist_km * km_to_px))


class MosaicObservation(ChmiObservation):
    """One precip-mosaic frame, evaluated like a ČHMÚ frame."""

    def __init__(self, frame_id: str, timestamp: int, lat: float, lon: float,
                 radar_config: dict, observation_type: str, map_tile: np.ndarray,
                 half_km: float, rate: np.ndarray, flags: np.ndarray,
                 colour: Optional[np.ndarray], rain_mmh: float, min_covered: float):
        size = int(radar_config.get('size', 512))
        super().__init__(b'', frame_id, timestamp, lat, lon, radar_config,
                         observation_type, map_tile,
                         half_px=max(1, size // 2), half_km=half_km)
        self.frame_id = frame_id
        self.rain_mmh = float(rain_mmh)
        self.min_covered = float(min_covered)
        self.rate = self._fit(rate)
        self.flags = self._fit(flags)
        self.colour = self._fit(colour) if colour is not None else None
        self.covered_fraction = 0.0
        self.available = False
        self.sources: List[str] = []

    def _fit(self, arr: np.ndarray) -> np.ndarray:
        """Resize to ``size`` × ``size`` (the service clamps the size)."""
        if arr.shape[:2] == (self._size, self._size):
            return arr
        return cv2.resize(arr, (self._size, self._size),
                          interpolation=cv2.INTER_NEAREST)

    def get_image(self):
        """Black where dry or uncovered, palette colour where it rains."""
        rain = np.isfinite(self.rate) & (np.nan_to_num(self.rate) >= self.rain_mmh)
        img = np.zeros((self._size, self._size, 3), dtype=np.uint8)
        img[rain] = _FALLBACK_BGR
        if self.colour is not None:
            painted = rain & (self.colour[:, :, 3] > 0)
            img[painted] = self.colour[:, :, :3][painted]
        for bgr in _NO_RAIN_BGR:
            clash = rain & np.all(img == bgr, axis=2)
            img[clash] = _FALLBACK_BGR
        self.img = img

    def _coverage(self) -> float:
        """Share of the rain square (robot position) with current data."""
        top, bottom, left, right = _square(self._size, self._half_km,
                                           float(self.radar_config.get('frame_rain', 1)))
        top, left = max(0, top), max(0, left)
        rate = self.rate[top:bottom, left:right]
        flags = self.flags[top:bottom, left:right]
        if rate.size == 0:
            return 0.0
        good = np.isfinite(rate) & ((flags & _FLAG_STALE) == 0)
        return float(good.sum()) / float(rate.size)

    def get_rain_status(self):
        """ČHMÚ frame evaluation, then ``UNAVAILABLE`` without coverage."""
        super().get_rain_status()
        self.covered_fraction = self._coverage()
        self.available = self.covered_fraction >= self.min_covered
        if not self.available:
            self.status = STATUS_UNAVAILABLE
            for img in (self.img_annotated, self.img_annotation):
                cv2.putText(img, 'NO RADAR DATA', (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, VALUES.color_white, 2)


class Mosaic(Chmi):
    """
    precip-mosaic source with the public interface of RainViewer / Chmi.
    Reuses the ČHMÚ pictures (OSM background, 2×2 forecast grid, clean
    rain-only RGBA) on top of the mosaic frames.
    """

    def __init__(self, lat: float, lon: float, radar_config: dict, use_map: int,
                 mosaic_config: Optional[dict] = None):
        # Chmi.__init__ is not called: its crop geometry is ČHMÚ-specific.
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.radar_config = radar_config
        self.use_map = use_map
        self.observations: List[MosaicObservation] = []
        self.nowcasts: List[MosaicObservation] = []
        self.generated = 0
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in (mosaic_config or {}).items() if v is not None})
        self.config = cfg
        self.url = str(cfg['url']).rstrip('/')
        self.timeout_s = float(cfg['timeout_s'])
        self.poll_s = float(cfg['poll_s'])
        self.rain_mmh = float(cfg['rain_mmh'])
        self.nowcast_max_min = int(cfg['nowcast_max_min'])
        self.min_covered = float(cfg['min_covered'])
        self._size = int(radar_config.get('size', 512))
        frame_warning_km = float(radar_config.get('frame_warning', 20))
        half_km = float(cfg['half_km'])
        self._half_km = half_km if half_km > 0 else 1.6 * frame_warning_km
        self._half_px = self._size // 2
        self.point: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self._online: Optional[bool] = None
        self._last_check = 0.0
        self._prefetched = False
        self._past_cache: Dict[str, MosaicObservation] = {}
        try:
            self.map_tile = self._build_map_tile()
        except Exception as exc:  # a missing basemap must not stop the alerts
            print(f"precip-mosaic: no map background ({exc})")
            self.map_tile = np.zeros((self._size, self._size, 4), dtype=np.uint8)

    # ── HTTP ────────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Dict[str, Any]) -> Tuple[bytes, Any]:
        # urlencode turns "+" into %2B: a raw "+" in nowcast+N would be
        # decoded as a space by the service.
        url = f"{self.url}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            detail = ''
            try:
                detail = json.loads(exc.read().decode('utf-8')).get('error', '')
            except Exception:
                pass
            raise MosaicError(f"{path}: HTTP {exc.code} {detail}".strip()) from None
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            reason = getattr(exc, 'reason', exc)
            raise MosaicError(f"{path}: {reason}") from None

    def _get_latest_id(self) -> Optional[str]:
        body, _ = self._get('/index.json', {})
        try:
            return json.loads(body.decode('utf-8')).get('latest')
        except (ValueError, AttributeError):
            raise MosaicError('/index.json: not JSON') from None

    def _get_point(self) -> Dict[str, Any]:
        body, _ = self._get('/point', {
            'lat': self.latitude, 'lon': self.longitude,
            'radius_km': float(self.radar_config.get('frame_warning', 20)),
            'past_min': 60})
        try:
            point = json.loads(body.decode('utf-8'))
        except ValueError:
            raise MosaicError('/point: not JSON') from None
        if not isinstance(point, dict) or 'error' in point:
            raise MosaicError(f"/point: {point.get('error') if isinstance(point, dict) else point}")
        return point

    def _get_area(self, frame: str, style: str) -> Tuple[bytes, Dict[str, Any]]:
        body, headers = self._get('/area.png', {
            'lat': self.latitude, 'lon': self.longitude, 'half_km': self._half_km,
            'size': self._size, 'frame': frame, 'style': style})
        try:
            meta = json.loads(headers.get('X-Precip-Meta') or '')
        except ValueError:
            raise MosaicError('/area.png: missing X-Precip-Meta') from None
        return body, meta

    def _fetch_frame(self, frame: str, expect_id: str, kind: str,
                     obs_type: str, colour: bool) -> MosaicObservation:
        """Download one frame (data, and the palette picture if ``colour``)."""
        data, meta = self._get_area(frame, 'data')
        if meta.get('frame') != expect_id or meta.get('kind') != kind:
            raise MosaicError(f"frame {frame} is {meta.get('frame')}/{meta.get('kind')}, "
                              f"expected {expect_id}/{kind} (time line moved)")
        rate, flags = decode_data_png(data)
        bgra = None
        if colour:
            png, meta_c = self._get_area(frame, 'rate')
            if meta_c.get('frame') != expect_id:
                raise MosaicError(f"frame {frame} moved to {meta_c.get('frame')}")
            bgra = decode_colour_png(png)
        obs = MosaicObservation(expect_id, parse_frame_id(expect_id),
                                self.latitude, self.longitude, self.radar_config,
                                obs_type, self.map_tile, self._half_km,
                                rate, flags, bgra, self.rain_mmh, self.min_covered)
        obs.sources = list(meta.get('sources') or [])
        obs.get_image()
        obs.get_rain_status()
        return obs

    # ── fetching ────────────────────────────────────────────────────────────

    def _fetch(self):
        """One complete refresh; raises and changes nothing on any error."""
        point = self._get_point()
        past_ids = sorted(p['time'] for p in point.get('past') or []
                          if p.get('kind') == 'obs' and p.get('time'))
        if not past_ids:
            raise MosaicError('no observed frame yet')
        past_ids = past_ids[-_KEEP_OBS:]
        now_id = past_ids[-1]
        latest_t = parse_frame_id(now_id)

        observations = []
        for frame_id in past_ids:
            obs = self._past_cache.get(frame_id)
            if obs is None or (frame_id == now_id and obs.colour is None):
                obs = self._fetch_frame(frame_id, frame_id, 'obs',
                                        'NOW' if frame_id == now_id else 'PAST',
                                        colour=frame_id == now_id)
                self._past_cache[frame_id] = obs
            observations.append(obs)
        self._past_cache = {o.frame_id: o for o in observations}

        nowcasts = []
        for item in point.get('nowcast') or []:
            frame_id = item.get('time')
            if not frame_id:
                continue
            ahead = (parse_frame_id(frame_id) - latest_t) // 60
            if ahead <= 0 or ahead > self.nowcast_max_min:
                continue
            nowcasts.append(self._fetch_frame(f"nowcast+{ahead}", frame_id, 'nowcast',
                                              'NOWCAST', colour=True))
        nowcasts.sort(key=lambda o: o.time)

        # Types as the ČHMÚ source sets them; the annotation shows the type,
        # so frames whose type changes are evaluated (drawn) again.
        for i, obs in enumerate(observations):
            new_type = 'NOW' if i == len(observations) - 1 else 'PAST'
            if obs.type != new_type:
                obs.type = new_type
                obs.get_rain_status()

        self.observations = observations
        self.nowcasts = nowcasts
        self.generated = latest_t
        self.point = point

    def _set_online(self, online: bool, error: Optional[str] = None):
        """Log only changes, the node polls every minute."""
        self.last_error = error
        if online != self._online:
            if online:
                print(f"precip-mosaic: OK ({self.url})")
            else:
                print(f"precip-mosaic: no data from {self.url}: {error}")
        self._online = online

    # ── public API (same as RainViewer / Chmi) ──────────────────────────────

    def get_map_data(self):
        """Fetch the time line and the new frames; keep the old ones on error."""
        self._last_check = time.time()
        try:
            self._fetch()
        except Exception as exc:
            self._set_online(False, str(exc))
            return
        self._set_online(True)

    def process_data(self):
        """Fetch, unless ``update_data`` has just done it."""
        if self._prefetched:
            self._prefetched = False
            return
        self.get_map_data()

    def update_data(self) -> bool:
        """
        Ask the service at most every ``poll_s`` seconds.

        :return: True only when a new observed frame arrived. While the
                 service is down nothing new is published, so the last
                 RainAlert ages (= no weather data) instead of turning dry.
        """
        now = time.time()
        # the node's loop also runs every 60 s: with the plain "< poll_s" a
        # loop a few ms early would skip a whole period, so the gate opens
        # POLL_SLACK_S before poll_s is reached
        if now - self._last_check < max(0.0, self.poll_s - POLL_SLACK_S):
            return False
        self._last_check = now
        if self.observations:
            # /index.json costs ~1 ms on the service, /point ~0.25 s: ask the
            # cheap one whether a new frame exists before the full refresh.
            try:
                if self._get_latest_id() == self.observations[-1].frame_id:
                    return False
            except MosaicError as exc:
                self._set_online(False, str(exc))
                return False
        before = self.generated
        self.get_map_data()
        changed = bool(self.observations) and self.generated != before
        self._prefetched = changed
        return changed

    def evaluate_data(self):
        """
        :return: (img_map, img_forecast, rain_alert, rain_now, rain_data)
        :raises MosaicError: no frame yet, or no radar coverage at the robot
                 in the NOW frame – the node then publishes nothing.
        """
        if not self.observations:
            raise MosaicError(f"no precip-mosaic data yet ({self.last_error})")
        now_obs = self.observations[-1]
        if not now_obs.available:
            raise MosaicError(f"no radar coverage at the robot in frame {now_obs.frame_id} "
                              f"({now_obs.covered_fraction:.0%} covered)")
        rain_data = self.get_rain_data()
        min_rain = float(self.radar_config.get('min_rain', 5.0))
        min_alert = float(self.radar_config.get('min_alert', 10.0))
        rain_now = now_obs.status_percentage_rain > min_rain
        rain_alert = any(o.status_percentage_rain > min_rain or
                         o.status_percentage_alert > min_alert
                         for o in self.nowcasts[:3])
        img_forecast = self._forecast_grid()
        img_map = now_obs.get_img_with_map()
        return img_map, img_forecast, rain_alert, rain_now, rain_data

    def get_rain_meta(self) -> Dict[str, Any]:
        """Extent of rain_img / map_img (keys of the ČHMÚ source) + origin."""
        size = self._size
        meta = {
            'center_lat': self.latitude,
            'center_lon': self.longitude,
            'half_km': float(self._half_km),
            'size_px': size,
            'km_per_px': (2.0 * self._half_km) / float(size) if size else 0.0,
            'frame_warning_km': float(self.radar_config.get('frame_warning', 20)),
            'source': 'mosaic',
        }
        if self.observations:
            meta['frame'] = self.observations[-1].frame_id
            meta['sources'] = list(getattr(self.observations[-1], 'sources', []))
        if self.point:
            meta['attribution'] = list(self.point.get('attribution') or [])
        return meta

    def print_rain_status(self):
        """Print all frame results to stdout."""
        status = self.get_rain_data()
        print(f"precip-mosaic — newest frame: {status.get('generated_date', '')}"
              f"{'' if self._online else f'  (offline: {self.last_error})'}")
        for obs in status['observations']:
            print('{:^20} {:^11} {:^7} [{:6.2f}% warn] [{:6.2f}% alert] [{:6.2f}% rain]'
                  .format(obs['date'], obs['final_status'], obs['type'],
                          obs['percent_warn'], obs['percent_alert'], obs['percent_rain']))

    def show_images(self):
        """Show all frames in sequence."""
        cv2.namedWindow("precip-mosaic", cv2.WINDOW_NORMAL)
        for obs in self.observations + self.nowcasts:
            cv2.imshow('precip-mosaic', obs.get_img_with_map())
            cv2.waitKey(2000)


__all__ = ['Mosaic', 'MosaicObservation', 'MosaicError', 'DEFAULTS',
           'decode_data_png', 'decode_colour_png', 'parse_frame_id',
           'STATUS_UNAVAILABLE']
