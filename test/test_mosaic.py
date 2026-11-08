#!/usr/bin/env python3
"""precip-mosaic radar source (weather_alert/mosaic.py). No ROS, no network.

A fake precip-mosaic HTTP server on an ephemeral localhost port answers
/index.json, /point and /area.png in the service's format (images encoded with
vitulus_precip.palette when that package is found next to this repo, else with
an equivalent local encoder). When vitulus_precip is available, the real
service is also started on its own recorded fixtures (Praha, public point).

Run:  python3 test/test_mosaic.py
"""
import calendar
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PKG, 'src'))

from weather_alert import radar_log  # noqa: E402
from weather_alert.chmi import Chmi  # noqa: E402
from weather_alert.mosaic import (Mosaic, MosaicError, STATUS_UNAVAILABLE,  # noqa: E402
                                  decode_colour_png, decode_data_png, parse_frame_id)

PRAHA = (50.08, 14.42)   # public point, never the garden
RADAR = {'frame_rain': 1, 'frame_alert': 9, 'frame_warning': 20, 'min_rain': 5.0,
         'min_alert': 10.0, 'min_warning': 30.0, 'size': 512, 'rain_threshold': 100}
CHMI_META_KEYS = {'center_lat', 'center_lon', 'half_km', 'size_px', 'km_per_px',
                  'frame_warning_km'}


def _precip_src():
    """Path of the vitulus_precip package sources, or None."""
    if os.environ.get('PRECIP_SRC') == 'none':     # force the local encoders
        return None
    here = os.path.abspath(__file__)
    cands = [os.environ.get('PRECIP_SRC', ''),
             '/ws/src/vitulus/vitulus_precip',
             os.path.join(_PKG, '..', 'vitulus_precip'),
             os.path.join(here.split('/wt/')[0], 'catkin_ws/src/vitulus/vitulus_precip')
             if '/wt/' in here else '']
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'src', 'vitulus_precip', 'palette.py')):
            return os.path.abspath(c)
    return None


PRECIP = _precip_src()
try:
    sys.path.insert(1, os.path.join(PRECIP or '', 'src'))
    from vitulus_precip import palette as PRECIP_PALETTE  # noqa: E402
except Exception:  # package not next to this repo, or PIL missing
    PRECIP_PALETTE = None


def iso(t):
    return time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(t))


# ── encoders in the service's format ───────────────────────────────────────

def encode_data(rate, flags=0, stale=None):
    """style=data: R,G = mm/h*100 u16 BE, B = type|flags, A = quality (0 = no data)."""
    if PRECIP_PALETTE is not None and stale is None:
        return PRECIP_PALETTE.encode_data_png(rate, np.ones(rate.shape, np.uint8),
                                              np.full(rate.shape, 0.8, np.float32), flags)
    covered = np.isfinite(rate)
    v = np.clip(np.round(np.where(covered, rate, 0) * 100.0), 0, 65535).astype(np.uint16)
    b = np.full(rate.shape, 1 | flags, np.uint8)
    if stale is not None:
        b[stale] |= 0x20
    bgra = np.zeros(rate.shape + (4,), np.uint8)
    bgra[..., 0] = np.where(covered, b, 0)
    bgra[..., 1] = v & 0xFF
    bgra[..., 2] = v >> 8
    bgra[..., 3] = np.where(covered, 204, 0)
    return cv2.imencode('.png', bgra)[1].tobytes()


def encode_rate(rate):
    """style=rate: palette colour, transparent where dry (< 0.1 mm/h) or uncovered."""
    if PRECIP_PALETTE is not None:
        return PRECIP_PALETTE.render_rate_png(rate)
    wet = np.isfinite(rate) & (np.nan_to_num(rate) >= 0.1)
    bgra = np.zeros(rate.shape + (4,), np.uint8)
    bgra[wet] = (230, 120, 25, 235)
    return cv2.imencode('.png', bgra)[1].tobytes()


# ── fake precip-mosaic service ─────────────────────────────────────────────

class Scenario:
    """Time line and rain field served by the fake service."""

    def __init__(self, latest_t):
        self.latest_t = latest_t
        self.nowcast_steps = (10, 20, 30, 40, 50, 60)
        self.blobs = {}        # frame id -> [(east_km, north_km, radius_km, mm/h)]
        self.holes = {}        # frame id -> [(east_km, north_km, radius_km)] without coverage
        self.stale = {}        # frame id -> [(east_km, north_km, radius_km)] stale data
        self.point_error = None
        self.delay_s = 0.0
        self.meta_frame_override = None
        self.requests = []
        self.lock = threading.Lock()

    def past_ids(self):
        return [iso(self.latest_t - m * 60) for m in range(60, -1, -5)]

    def nowcast_ids(self):
        return [iso(self.latest_t + m * 60) for m in self.nowcast_steps]

    def resolve(self, frame):
        if frame in ('latest', 'now'):
            return iso(self.latest_t), 'obs'
        if frame.startswith('nowcast+'):
            fid = iso(self.latest_t + int(frame.split('+', 1)[1]) * 60)
            return (fid, 'nowcast') if fid in self.nowcast_ids() else (None, None)
        if frame in self.past_ids():
            return frame, 'obs'
        if frame in self.nowcast_ids():
            return frame, 'nowcast'
        return None, None

    def field(self, fid, half_km, size):
        """Rate grid like PrecipService._local_grid: row 0 north, linear km."""
        k = np.linspace(-half_km, half_km, size)
        x, y = np.meshgrid(k, -k)
        rate = np.zeros((size, size), np.float32)
        for ex, ny, r, mmh in self.blobs.get(fid, []):
            rate[np.hypot(x - ex, y - ny) <= r] = mmh
        for ex, ny, r in self.holes.get(fid, []):
            rate[np.hypot(x - ex, y - ny) <= r] = np.nan
        stale = None
        if fid in self.stale:
            stale = np.zeros((size, size), bool)
            for ex, ny, r in self.stale[fid]:
                stale |= np.hypot(x - ex, y - ny) <= r
        return rate, stale


class ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(sc):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype='application/json', headers=None):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the client gave up (timeout test)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            with sc.lock:
                sc.requests.append((u.path, q))
            if sc.delay_s:
                time.sleep(sc.delay_s)
            if u.path == '/index.json':
                return self.send(200, {'version': 1, 'latest': iso(sc.latest_t)})
            if u.path == '/point':
                if sc.point_error:
                    return self.send(400, {'error': sc.point_error})
                past = [{'time': f, 'kind': 'obs', 'rate_mmh': 0.0, 'ptype': 'none',
                         'quality': 0.85, 'source': 'chmi'} for f in sc.past_ids()]
                nowcast = [{'time': f, 'kind': 'nowcast', 'rate_mmh': 0.0, 'ptype': 'none',
                            'quality': 0.6, 'source': 'chmi'} for f in sc.nowcast_ids()]
                return self.send(200, {'lat': float(q['lat']), 'lon': float(q['lon']),
                                       'now': past[-1], 'past': past, 'nowcast': nowcast,
                                       'around': {'radius_km': float(q['radius_km'])},
                                       'outlook': {}, 'sources': ['chmi'],
                                       'attribution': ['(c) test radar']})
            if u.path == '/area.png':
                fid, kind = sc.resolve(q.get('frame', 'latest'))
                if fid is None:
                    return self.send(400, {'error': 'unknown frame'})
                half_km, size = float(q['half_km']), int(q['size'])
                rate, stale = sc.field(fid, half_km, size)
                style = q.get('style', 'rate')
                png = encode_data(rate, 0x10 if kind == 'nowcast' else 0, stale) \
                    if style == 'data' else encode_rate(rate)
                meta = {'center': {'lat': float(q['lat']), 'lon': float(q['lon'])},
                        'half_km': half_km, 'km_per_px': 2 * half_km / size, 'size': size,
                        'frame': sc.meta_frame_override or fid, 'kind': kind,
                        'sources': ['chmi']}
                return self.send(200, png, 'image/png', {'X-Precip-Meta': json.dumps(meta)})
            return self.send(404, {'error': 'not found'})
    return Handler


class FakeService:
    def __init__(self, scenario, port=0):
        self.httpd = ThreadingServer(('127.0.0.1', port), make_handler(scenario))
        self.port = self.httpd.server_address[1]
        self.url = 'http://127.0.0.1:%d' % self.port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port():
    s = HTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
    port = s.server_address[1]
    s.server_close()
    return port


T0 = calendar.timegm((2026, 11, 7, 12, 0, 0))


class Base(unittest.TestCase):
    def setUp(self):
        self.sc = Scenario(T0)
        self.svc = FakeService(self.sc)

    def tearDown(self):
        self.svc.stop()

    def source(self, **cfg):
        cfg.setdefault('url', self.svc.url)
        cfg.setdefault('timeout_s', 5)
        m = Mosaic(PRAHA[0], PRAHA[1], dict(RADAR), 0, cfg)
        m.process_data()
        return m

    def obs(self, m, fid):
        return next(o for o in m.observations + m.nowcasts if o.frame_id == fid)


# ── tests ──────────────────────────────────────────────────────────────────

class Decoding(unittest.TestCase):
    def test_data_png_values_coverage_flags(self):
        rate = np.array([[0.0, 0.15, 5.0, 200.0, np.nan]], np.float32)
        stale = np.array([[False, False, True, False, False]])
        r, flags = decode_data_png(encode_data(rate, 0x10, stale))
        np.testing.assert_allclose(r[0, :4], rate[0, :4], atol=0.006)
        self.assertTrue(np.isnan(r[0, 4]))
        self.assertTrue(flags[0, 2] & 0x20 and not flags[0, 1] & 0x20)
        self.assertTrue(all(flags[0, :4] & 0x10))

    @unittest.skipIf(PRECIP_PALETTE is None, 'vitulus_precip not found')
    def test_real_palette_formats(self):
        rate = np.array([[np.nan, 0.0, 0.15, 5.0, 200.0]], np.float32)
        r, _ = decode_data_png(PRECIP_PALETTE.encode_data_png(rate))
        np.testing.assert_allclose(r[0, 1:], rate[0, 1:], atol=0.006)
        self.assertTrue(np.isnan(r[0, 0]))
        bgra = decode_colour_png(PRECIP_PALETTE.render_rate_png(rate))   # indexed + tRNS
        self.assertEqual(bgra.shape, (1, 5, 4))
        self.assertEqual(list(bgra[0, :, 3] > 0), [False, False, True, True, True])

    def test_frame_id(self):
        self.assertEqual(parse_frame_id('2026-11-07T12:00Z'), T0)
        self.assertEqual(iso(parse_frame_id('2026-11-08T09:35Z')), '2026-11-08T09:35Z')


class Frames(Base):
    def test_dry_full_timeline(self):
        m = self.source()
        self.assertIsNone(m.last_error)
        self.assertEqual(len(m.observations), 13)
        self.assertEqual(len(m.nowcasts), 6)
        self.assertEqual([o.type for o in m.observations], ['PAST'] * 12 + ['NOW'])
        self.assertEqual(m.observations[-1].time, T0)
        self.assertEqual([o.time - T0 for o in m.nowcasts], [600 * i for i in range(1, 7)])
        self.assertTrue(all(o.status == 'OK' for o in m.observations + m.nowcasts))
        img_map, img_fc, rain_alert, rain_now, rain_data = m.evaluate_data()
        self.assertEqual(img_map.shape, (512, 512, 3))
        self.assertEqual(img_fc.shape, (512, 512, 3))
        self.assertFalse(rain_alert or rain_now)
        self.assertEqual(rain_data['generated'], T0)
        rgba = m.get_rain_only_rgba()
        self.assertEqual((rgba.shape, rgba.dtype, int(rgba[..., 3].max())), ((512, 512, 4), np.uint8, 0))

    def test_rain_data_same_shape_as_chmi(self):
        m = self.source()
        rd = m.get_rain_data()
        chmi_keys = set(Chmi.get_rain_data(type('E', (), {
            'observations': [], 'nowcasts': [], 'generated': 0,
            'latitude': 0.0, 'longitude': 0.0})()).keys())
        self.assertEqual(set(rd.keys()), chmi_keys)
        self.assertEqual(set(rd['observations'][0].keys()),
                         {'time', 'final_status', 'date', 'type', 'percent_warn',
                          'percent_alert', 'percent_rain'})
        sample = radar_log.sample_from(rd, False, False, received=T0 + 60)
        self.assertEqual(sample['t'], T0)
        self.assertEqual(sorted(p[0] for p in sample['past']), [-60, -50, -40, -30, -20, -10])
        self.assertEqual(sorted(p[0] for p in sample['nc']), [10, 20, 30])

    def test_rain_at_robot_now(self):
        self.sc.blobs[iso(T0)] = [(0.0, 0.0, 3.0, 2.0)]
        m = self.source()
        now = m.observations[-1]
        self.assertEqual(now.status, 'RAIN')
        self.assertEqual(now.status_percentage_rain, 100.0)
        self.assertGreater(now.status_percentage_alert, 5.0)
        self.assertLess(now.status_percentage_warn, now.status_percentage_alert)
        img_map, _, _, rain_now, _ = m.evaluate_data()
        self.assertTrue(rain_now)
        rgba = m.get_rain_only_rgba()
        self.assertEqual(int(rgba[256, 256, 3]), 255)        # rain: opaque
        self.assertEqual(tuple(rgba[5, 5]), (0, 0, 0, 0))      # dry: transparent black
        self.assertTrue(rgba[256, 256, :3].any())
        self.assertEqual(m.observations[-2].status, 'OK')

    def test_frame_squares_in_km(self):
        # 0.5 km blob 14 km east: inside the 20 km square only; 6 km north: 9 km square
        self.sc.blobs[iso(T0 - 600)] = [(14.0, 0.0, 0.5, 5.0)]
        self.sc.blobs[iso(T0 - 1200)] = [(0.0, 6.0, 0.5, 5.0)]
        m = self.source()
        warn, alert = self.obs(m, iso(T0 - 600)), self.obs(m, iso(T0 - 1200))
        self.assertEqual((warn.status, warn.status_percentage_alert), ('WARN', 0.0))
        self.assertGreater(warn.status_percentage_warn, 0.0)
        self.assertEqual((alert.status, alert.status_percentage_rain), ('ALERT', 0.0))
        # π·0.5² km² of (2·9 km)² ≈ 0.24 %
        self.assertAlmostEqual(alert.status_percentage_alert, 100 * np.pi * 0.25 / 324, delta=0.1)

    def test_nowcast_rain_and_alert(self):
        self.sc.blobs[iso(T0 + 1200)] = [(0.0, 0.0, 6.0, 4.0)]
        m = self.source()
        nc = self.obs(m, iso(T0 + 1200))
        self.assertEqual((nc.type, nc.status), ('NOWCAST', 'RAIN'))
        _, img_fc, rain_alert, rain_now, _ = m.evaluate_data()
        self.assertTrue(rain_alert)
        self.assertFalse(rain_now)
        nowcast_frames = [q['frame'] for p, q in self.sc.requests
                          if p == '/area.png' and q['frame'].startswith('nowcast')]
        self.assertIn('nowcast+20', nowcast_frames)   # "+" survived the query string

    def test_rain_threshold(self):
        self.sc.blobs[iso(T0)] = [(0.0, 0.0, 3.0, 0.05)]
        self.assertEqual(self.source().observations[-1].status, 'OK')
        m = self.source(rain_mmh=0.02)
        self.assertEqual(m.observations[-1].status, 'RAIN')
        rgba = m.get_rain_only_rgba()   # below the palette: fallback colour, still opaque
        self.assertEqual(int(rgba[256, 256, 3]), 255)

    def test_rain_meta(self):
        m = self.source()
        meta = m.get_rain_meta()
        self.assertTrue(CHMI_META_KEYS <= set(meta))
        self.assertEqual((meta['half_km'], meta['size_px'], meta['km_per_px']), (32.0, 512, 0.125))
        self.assertEqual((meta['source'], meta['frame']), ('mosaic', iso(T0)))
        self.assertEqual(meta['attribution'], ['(c) test radar'])
        json.dumps(meta)
        self.assertEqual(self.source(half_km=40).get_rain_meta()['half_km'], 40.0)


class Unknown(Base):
    def test_uncovered_past_frame_is_unavailable(self):
        self.sc.holes[iso(T0 - 900)] = [(0.0, 0.0, 5.0)]
        self.sc.blobs[iso(T0 - 900)] = [(10.0, 0.0, 2.0, 3.0)]
        m = self.source()
        self.assertEqual(self.obs(m, iso(T0 - 900)).status, STATUS_UNAVAILABLE)
        m.evaluate_data()   # NOW is covered: publishes

    def test_uncovered_now_publishes_nothing(self):
        self.sc.holes[iso(T0)] = [(0.0, 0.0, 30.0)]
        m = self.source()
        self.assertEqual(m.observations[-1].status, STATUS_UNAVAILABLE)
        with self.assertRaises(MosaicError):
            m.evaluate_data()

    def test_stale_now_publishes_nothing(self):
        self.sc.stale[iso(T0)] = [(0.0, 0.0, 30.0)]
        m = self.source()
        self.assertEqual(m.observations[-1].status, STATUS_UNAVAILABLE)
        with self.assertRaises(MosaicError):
            m.evaluate_data()

    def test_point_outside_area(self):
        self.sc.point_error = 'point outside the configured area'
        m = self.source()
        self.assertEqual(m.observations, [])
        self.assertIn('outside the configured area', m.last_error)
        with self.assertRaises(MosaicError):
            m.evaluate_data()

    def test_timeline_moved_changes_nothing(self):
        m = self.source()
        self.sc.latest_t = T0 + 300
        self.sc.meta_frame_override = iso(T0 - 7200)
        m._last_check = 0
        self.assertFalse(m.update_data())
        self.assertIn('time line moved', m.last_error)
        self.assertEqual(m.observations[-1].time, T0)
        self.assertEqual(len(m.nowcasts), 6)


class Outage(unittest.TestCase):
    def test_service_down_at_start(self):
        port = free_port()
        t = time.time()
        m = Mosaic(PRAHA[0], PRAHA[1], dict(RADAR), 0,
                   {'url': 'http://127.0.0.1:%d' % port, 'timeout_s': 2})
        m.process_data()
        self.assertEqual((m.observations, m.nowcasts), ([], []))
        self.assertTrue(m.last_error)
        with self.assertRaises(MosaicError):
            m.evaluate_data()
        m._last_check = 0
        self.assertFalse(m.update_data())
        self.assertLess(time.time() - t, 5)
        # the service comes up: next poll fetches and reports new data once
        sc = Scenario(T0)
        svc = FakeService(sc, port)
        try:
            m._last_check = 0
            self.assertTrue(m.update_data())
            n = len(sc.requests)
            m.process_data()                  # the node calls it right after
            self.assertEqual(len(sc.requests), n)
            self.assertEqual(len(m.observations), 13)
            self.assertIsNone(m.last_error)
        finally:
            svc.stop()

    def test_outage_keeps_last_frames_and_recovers(self):
        sc = Scenario(T0)
        svc = FakeService(sc)
        port = svc.port
        m = Mosaic(PRAHA[0], PRAHA[1], dict(RADAR), 0, {'url': svc.url, 'timeout_s': 2})
        m.process_data()
        svc.stop()
        m._last_check = 0
        self.assertFalse(m.update_data())
        self.assertTrue(m.last_error)
        self.assertEqual(m.observations[-1].time, T0)
        m.evaluate_data()                      # old data still evaluates (node won't call it)
        sc.latest_t = T0 + 300
        svc = FakeService(sc, port)
        try:
            m._last_check = 0
            self.assertTrue(m.update_data())
            self.assertEqual(m.observations[-1].time, T0 + 300)
            self.assertEqual(len(m.observations), 13)
        finally:
            svc.stop()

    def test_timeout(self):
        sc = Scenario(T0)
        svc = FakeService(sc)
        try:
            sc.delay_s = 1.5
            t = time.time()
            m = Mosaic(PRAHA[0], PRAHA[1], dict(RADAR), 0, {'url': svc.url, 'timeout_s': 0.3})
            m.process_data()
            self.assertLess(time.time() - t, 1.4)
            self.assertEqual(m.observations, [])
            self.assertIn('timed out', m.last_error)
        finally:
            sc.delay_s = 0
            svc.stop()


class Polling(Base):
    def test_cheap_poll_and_incremental_fetch(self):
        m = self.source()
        first = [(p, q.get('frame'), q.get('style')) for p, q in self.sc.requests]
        self.assertEqual(sum(1 for p, _, _ in first if p == '/point'), 1)
        # 13 past (data) + 6 nowcast (data) + NOW and 6 nowcast (rate)
        self.assertEqual(sum(1 for p, _, _ in first if p == '/area.png'), 13 + 6 + 7)
        self.sc.requests.clear()
        self.assertFalse(m.update_data())             # within poll_s: no request
        self.assertEqual(self.sc.requests, [])
        # the node loop (60 s) comes a little early: still asked, not skipped
        m._last_check = time.time() - (m.poll_s - 2)
        self.assertFalse(m.update_data())             # nothing new: only /index.json
        self.assertEqual([p for p, _ in self.sc.requests], ['/index.json'])
        self.sc.requests.clear()
        m._last_check = time.time() - (m.poll_s - 10)  # clearly too early: nothing
        self.assertFalse(m.update_data())
        self.assertEqual(self.sc.requests, [])
        m._last_check = 0
        self.assertFalse(m.update_data())             # nothing new: only /index.json
        self.assertEqual([p for p, _ in self.sc.requests], ['/index.json'])
        self.sc.requests.clear()
        self.sc.latest_t = T0 + 300
        m._last_check = 0
        self.assertTrue(m.update_data())
        past = [q for p, q in self.sc.requests
                if p == '/area.png' and not q['frame'].startswith('nowcast')]
        self.assertEqual(sorted(q['style'] for q in past), ['data', 'rate'])
        self.assertEqual({q['frame'] for q in past}, {iso(T0 + 300)})
        self.assertEqual([o.type for o in m.observations[-2:]], ['PAST', 'NOW'])
        self.assertEqual(m.observations[0].time, T0 + 300 - 3600)


@unittest.skipIf(PRECIP_PALETTE is None, 'vitulus_precip not found')
class RealService(unittest.TestCase):
    """The real precip-mosaic server on its recorded fixtures (Praha)."""

    @classmethod
    def setUpClass(cls):
        try:
            sys.path.insert(0, os.path.join(PRECIP, 'test'))
            import helpers  # vitulus_precip/test/helpers.py
            from vitulus_precip.server import ThreadingHTTPServer, make_handler
            import h5py  # noqa: F401  (ODIM sources)
        except Exception as exc:
            raise unittest.SkipTest('vitulus_precip test helpers unavailable: %s' % exc)
        cls.helpers = helpers
        cls.svc = helpers.make_service(area={'lat': PRAHA[0], 'lon': PRAHA[1], 'radius_km': 100,
                                             'min_zoom': 5, 'max_zoom': 7})
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(cls.svc))
        cls.url = 'http://127.0.0.1:%d' % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.helpers.cleanup(cls.svc)

    def test_real_format(self):
        m = Mosaic(PRAHA[0], PRAHA[1], dict(RADAR), 0, {'url': self.url, 'timeout_s': 30})
        m.process_data()
        self.assertIsNone(m.last_error)
        self.assertTrue(m.observations)
        self.assertEqual(m.observations[-1].type, 'NOW')
        self.assertTrue(m.nowcasts, 'ČHMÚ fixture carries a nowcast')
        self.assertTrue(all(0 < o.time - m.generated <= 3600 for o in m.nowcasts))
        img_map, img_fc, _, _, rain_data = m.evaluate_data()
        self.assertEqual(img_map.shape, (512, 512, 3))
        self.assertEqual(len(rain_data['observations']), len(m.observations) + len(m.nowcasts))
        for o in m.observations + m.nowcasts:
            self.assertIn(o.status, ('OK', 'WARN', 'ALERT', 'RAIN', STATUS_UNAVAILABLE))
        meta = m.get_rain_meta()
        self.assertTrue(meta['sources'])
        self.assertTrue(meta['attribution'])
        print('\n  real service: %d obs + %d nowcast, NOW %s %s, sources %s'
              % (len(m.observations), len(m.nowcasts), m.observations[-1].frame_id,
                 m.observations[-1].status, meta['sources']))


if __name__ == '__main__':
    print('vitulus_precip palette: %s' % (PRECIP or 'not found (local encoder)'))
    unittest.main(verbosity=2)
