#!/usr/bin/env python3
"""Radar record, Open-Meteo parsing/storage and history.summary(). No ROS, no network.

Run:  python3 test/test_weather_history.py
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

os.environ['TZ'] = 'Europe/Prague'
time.tzset()
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PKG, 'src'))

from weather_alert import history, openmeteo, radar_log, storage  # noqa: E402

SAMPLE = os.path.join(_PKG, 'test', 'data', 'openmeteo_forecast_sample.json')


def obs(t, typ, rain=0.0, alert=0.0, warn=0.0, status='OK'):
    return {'time': t, 'type': typ, 'final_status': status,
            'percent_rain': rain, 'percent_alert': alert, 'percent_warn': warn}


def rain_data(now_t, rain_now=0.0, nowcast_rain=0.0):
    o = [obs(now_t - m * 60, 'PAST', rain=1.0) for m in (60, 50, 40, 30, 20, 10)]
    o.append(obs(now_t, 'NOW', rain=rain_now, status='RAIN' if rain_now else 'OK'))
    o += [obs(now_t + m * 60, 'NOWCAST', rain=nowcast_rain) for m in (10, 20, 30, 40)]
    return {'generated': now_t, 'observations': o}


def row(t, **kw):
    r = {key: None for _, key in openmeteo.HOURLY}
    r.update(time=t, precip_mm=0.0, precip_prob_pct=0, temp_c=10.0, et0_mm=0.05)
    r.update(kw)
    return r


class Tmp(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix='weather_test_')

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)


class Radar(Tmp):
    T0 = int(dt.datetime(2026, 11, 7, 12, 0).timestamp())

    def test_sample_is_compact(self):
        s = radar_log.sample_from(rain_data(self.T0, rain_now=12.345, nowcast_rain=7),
                                  True, True, received=self.T0 + 300)
        self.assertEqual((s['t'], s['rx'], s['st'], s['rain'], s['alert']),
                         (self.T0, self.T0 + 300, 'RAIN', 1, 1))
        self.assertEqual(s['now'], [12.3, 0.0, 0.0])
        self.assertEqual([p[0] for p in s['past']], [-10, -20, -30, -40, -50, -60])
        self.assertEqual(s['nc'], [[10, 7.0, 0.0, 0.0], [20, 7.0, 0.0, 0.0], [30, 7.0, 0.0, 0.0]])
        self.assertIsNone(radar_log.sample_from({'observations': []}, False, False))

    def test_record_dedup_and_day_summary(self):
        log = radar_log.RadarLog(self.base)
        for i in range(12):  # an hour, 5-min frames, rain in frames 3..5
            t = self.T0 + i * 300
            rn = 3 <= i <= 5
            s = radar_log.sample_from(rain_data(t, rain_now=20.0 + i if rn else 0.0), rn, False)
            self.assertTrue(log.record(s))
            self.assertFalse(log.record(s))  # same frame again
        path = log.day_path(dt.date(2026, 11, 7))
        with open(path) as f:
            self.assertEqual(len(f.readlines()), 12)
        day = log.daily()['2026-11-07']
        self.assertEqual(day['samples'], 12)
        self.assertEqual(day['rain_minutes'], 15.0)
        self.assertEqual(day['max_rain_pct'], 25.0)
        self.assertEqual(day['last_rain'], self.T0 + 5 * 300)
        self.assertEqual(log.last_rain(self.T0 + 3600), self.T0 + 5 * 300)
        # a fresh object (node restart) still knows the last frame
        self.assertFalse(radar_log.RadarLog(self.base).record(s))

    def test_torn_line_skipped(self):
        log = radar_log.RadarLog(self.base)
        log.record(radar_log.sample_from(rain_data(self.T0), False, False))
        with open(log.day_path(dt.date(2026, 11, 7)), 'a') as f:
            f.write('{"t": 17')
        self.assertEqual(len(log.read_day(dt.date(2026, 11, 7))), 1)

    def test_gap_is_not_rain(self):
        samples = [{'t': self.T0, 'rain': 1, 'now': [50, 0, 0]},
                   {'t': self.T0 + 3 * 3600, 'rain': 0, 'now': [0, 0, 0]}]
        self.assertEqual(radar_log.day_summary(samples)['rain_minutes'], 10.0)


class OpenMeteo(Tmp):
    def setUp(self):
        super().setUp()
        with open(SAMPLE) as f:
            self.payload = json.load(f)
        self.fetched = self.payload['current']['time']

    def test_parse_sample(self):
        rows = openmeteo.parse_hourly(self.payload)
        self.assertEqual(len(rows), 120)
        self.assertEqual(rows[1]['time'] - rows[0]['time'], 3600)
        self.assertEqual(set(rows[0]), {'time'} | {k for _, k in openmeteo.HOURLY})
        self.assertTrue(all(isinstance(r['temp_c'], float) for r in rows))
        cur = openmeteo.parse_current(self.payload)
        self.assertEqual(cur['time'], self.fetched)
        self.assertIsInstance(cur['temp_c'], float)
        days = openmeteo.daily_list(rows)
        self.assertEqual(len(days), 5)
        self.assertEqual(days[0]['hours'], 24)
        self.assertIsNotNone(days[0]['et0_mm'])

    def test_nulls_and_garbage(self):
        p = {'hourly': {'time': [3600, 'x', 7200], 'temperature_2m': [1.5, 2, None],
                        'precipitation': [float('nan'), 0, True]}}
        rows = openmeteo.parse_hourly(p)
        self.assertEqual([r['time'] for r in rows], [3600, 7200])
        self.assertIsNone(rows[0]['precip_mm'])
        self.assertIsNone(rows[1]['temp_c'])
        self.assertIsNone(rows[1]['precip_mm'])

    def test_save_archive_and_views(self):
        store = openmeteo.OpenMeteoStore(self.base)
        store.save(self.payload, self.fetched, 50.0187, 14.9033, '10_2026')
        today = storage.local_date(self.fetched)
        for back in (1, 2):
            doc = store.archive_day(today - dt.timedelta(days=back))
            self.assertTrue(doc['final'])
            self.assertEqual(len(doc['hourly']), 24)
        doc = store.archive_day(today)
        self.assertFalse(doc['final'])
        self.assertTrue(all(r['time'] <= self.fetched for r in doc['hourly']))
        self.assertIsNone(store.archive_day(today + dt.timedelta(days=1)))
        self.assertEqual(store.missing_days(today, 2), [])
        self.assertEqual(len(store.missing_days(today, 5)), 3)
        fc = openmeteo.forecast_view(store.latest(), self.fetched + 60, 48, status=store.status())
        self.assertTrue(fc['ok'])
        self.assertFalse(fc['stale'])
        self.assertEqual(len(fc['hourly']), 48)
        self.assertEqual(fc['age_s'], 60)
        stale = openmeteo.forecast_view(store.latest(), self.fetched + 5 * 3600, 48)
        self.assertTrue(stale['stale'])
        self.assertEqual(len(store.hourly(self.fetched - 86400, self.fetched)), 24)

    def test_offline(self):
        with mock.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('down')):
            with self.assertRaises(RuntimeError):
                openmeteo.fetch(50.0, 14.9, timeout=1)
        store = openmeteo.OpenMeteoStore(self.base)
        store.save(self.payload, self.fetched, 50.0, 14.9)
        st = store.set_status(False, self.fetched + 3600, 'down')
        self.assertEqual((st['last_ok'], st['last_error'], st['last_success']),
                         (False, 'down', self.fetched))
        self.assertIsNotNone(store.latest())  # old data stay
        fc = openmeteo.forecast_view(store.latest(), self.fetched + 4 * 3600, status=store.status())
        self.assertEqual((fc['ok'], fc['stale'], fc['last_error']), (True, True, 'down'))
        self.assertFalse(openmeteo.forecast_view(None, self.fetched)['ok'])

    def test_url(self):
        p = openmeteo.build_params(50.01870316, 14.90331712, 30, 3)
        self.assertEqual((p['latitude'], p['past_days'], p['timeformat']),
                         ('50.01870', 30, 'unixtime'))
        self.assertIn('et0_fao_evapotranspiration', p['hourly'])
        self.assertIn('precipitation_probability', p['hourly'])


class Summary(Tmp):
    NOW = int(dt.datetime(2026, 11, 7, 12, 0).timestamp())

    def make(self, rows, current=None, fetched=None):
        store = openmeteo.OpenMeteoStore(self.base)
        fetched = fetched or self.NOW - 600
        payload = {'hourly': {'time': [r['time'] for r in rows]},
                   'current': current or {}}
        for src, key in openmeteo.HOURLY:
            payload['hourly'][src] = [r[key] for r in rows]
        store.save(payload, fetched, 50.0, 14.9, 'test')
        return store

    def test_empty(self):
        s = history.summary(self.NOW, base=self.base)
        self.assertIsNone(s['hours_since_rain'])
        self.assertIsNone(s['temp_now'])
        self.assertIsNone(s['forecast_window_dry_h'])
        self.assertFalse(s['rain_last_h'])
        self.assertTrue(s['stale']['forecast'] and s['stale']['radar'])
        self.assertEqual(s['no_rain_within_days'], 30)

    def test_rain_and_forecast(self):
        h = 3600
        rows = [row(self.NOW + k * h) for k in range(-8 * 24, 49)]
        by = {r['time']: r for r in rows}
        by[self.NOW - 3 * h]['precip_mm'] = 1.5       # 3 h ago
        by[self.NOW - 30 * h]['precip_mm'] = 4.0      # yesterday
        by[self.NOW - 6 * 24 * h]['precip_mm'] = 2.0  # 6 days ago
        by[self.NOW - 9 * 24 * h] = row(self.NOW - 9 * 24 * h)  # outside 7 d
        by[self.NOW + 5 * h]['precip_prob_pct'] = 60  # rain likely 4-5 h from now
        by[self.NOW]['temp_c'] = 4.2
        by[self.NOW + 10 * h]['temp_c'] = -1.0
        self.make(sorted(by.values(), key=lambda r: r['time']))
        s = history.summary(self.NOW, base=self.base)
        self.assertEqual(s['hours_since_rain'], 3.0)
        self.assertEqual(s['last_rain_source'], 'open-meteo')
        self.assertEqual(s['rain_24h_mm'], 1.5)
        self.assertEqual(s['rain_7d_mm'], 7.5)
        self.assertFalse(s['rain_last_h'])
        self.assertEqual(s['temp_now'], 4.2)
        self.assertEqual(s['forecast_rain_next_h'], 0.0)
        self.assertEqual(s['forecast_window_dry_h'], 4.0)
        self.assertEqual(s['temp_min_next_24h'], -1.0)
        self.assertFalse(s['stale']['forecast'])
        self.assertEqual(s['forecast_age_s'], 600)
        self.assertAlmostEqual(s['et0_7d_mm'], 0.05 * 7 * 24, places=2)
        self.assertAlmostEqual(s['water_balance_7d_mm'], 7.5 - 8.4, places=2)

    def test_radar_wins_when_newer_and_blocks_window(self):
        h = 3600
        self.make([row(self.NOW + k * h, precip_mm=(2.0 if k == -5 else 0.0))
                   for k in range(-24, 49)])
        log = radar_log.RadarLog(self.base)
        for i in range(10):
            t = self.NOW - 3000 + i * 300
            rn = i in (2, 3)
            log.record(radar_log.sample_from(rain_data(t, rain_now=30 if rn else 0), rn, False))
        s = history.summary(self.NOW, base=self.base)
        self.assertEqual(s['last_rain_source'], 'radar')
        self.assertEqual(s['hours_since_rain'], round((3000 - 900) / 3600.0, 2))
        self.assertTrue(s['rain_last_h'])
        self.assertEqual(s['radar_rain_min_1h'], 10.0)
        self.assertFalse(s['stale']['radar'])
        self.assertEqual(s['forecast_window_dry_h'], 48.0)
        self.assertTrue(s['forecast_window_capped'])
        # radar nowcast alert now -> no dry window
        log.record(radar_log.sample_from(rain_data(self.NOW + 300), False, True))
        s = history.summary(self.NOW + 400, base=self.base)
        self.assertEqual(s['forecast_window_dry_h'], 0.0)
        self.assertTrue(s['radar']['rain_alert'])

    def test_history_doc(self):
        h = 3600
        self.make([row(self.NOW + k * h) for k in range(-72, 49)])
        doc = history.history(self.NOW, base=self.base, days=30)
        dates = [d['date'] for d in doc['days']]
        self.assertEqual(dates, ['2026-11-04', '2026-11-05', '2026-11-06', '2026-11-07'])
        self.assertTrue(doc['days'][1]['openmeteo']['final'])
        self.assertFalse(doc['days'][-1]['openmeteo']['final'])

    def test_site_location(self):
        root = os.path.join(self.base, 'mapping_v3')
        os.makedirs(os.path.join(root, 's1'))
        self.assertIsNone(storage.site_location(root))
        with open(os.path.join(root, '.last_served'), 'w') as f:
            json.dump({'site': 's1'}, f)
        with open(os.path.join(root, 's1', 'datum.yaml'), 'w') as f:
            f.write('origin_lat: 50.01870316\norigin_lon: 14.90331712\n')
        self.assertEqual(storage.site_location(root), (50.01870316, 14.90331712, 's1'))



class ChmiTime(unittest.TestCase):
    """ČHMÚ file names are UTC (not local time)."""

    def test_names_are_utc(self):
        from weather_alert import chmi
        t = chmi._parse_timestamp_from_name('pacz2gmaps3.z_cappi020.20261108.0005.0.png')
        self.assertEqual(t, int(dt.datetime(2026, 11, 8, 0, 5, tzinfo=dt.timezone.utc).timestamp()))
        self.assertEqual(dt.datetime.fromtimestamp(t).strftime('%H:%M'), '01:05')  # CET
        summer = chmi._parse_timestamp_from_name('pacz2gmaps3.fct_z_cappi020.20260706.1200.10.png')
        self.assertEqual(dt.datetime.fromtimestamp(summer).strftime('%H:%M'), '14:00')  # CEST
        # spacing between frames (what the rain status picks by) is unchanged
        a = chmi._parse_timestamp_from_name('x.20261108.0005.0.png')
        b = chmi._parse_timestamp_from_name('x.20261108.0015.0.png')
        self.assertEqual(b - a, 600)
        self.assertEqual(chmi._parse_timestamp_from_name('nothing.png'), 0)


class Robustness(Tmp):
    NOW = int(dt.datetime(2026, 11, 7, 12, 0).timestamp())

    def test_file_modes_0644(self):
        old = os.umask(0o022)
        try:
            storage.write_json(os.path.join(self.base, 'a', 'x.json'), {'a': 1})
            storage.append_line(os.path.join(self.base, 'a', 'y.jsonl'), '{}')
        finally:
            os.umask(old)
        for name in ('x.json', 'y.jsonl'):
            self.assertEqual(os.stat(os.path.join(self.base, 'a', name)).st_mode & 0o777, 0o644)

    def test_radar_dedup_from_file_and_torn_tail(self):
        a, b = radar_log.RadarLog(self.base), radar_log.RadarLog(self.base)
        s1 = radar_log.sample_from(rain_data(self.NOW), False, False)
        s2 = radar_log.sample_from(rain_data(self.NOW + 300), False, False)
        self.assertTrue(a.record(s1))
        self.assertFalse(b.record(s1))  # other writer: sees the file
        path = a.day_path(dt.date(2026, 11, 7))
        with open(path, 'a') as f:
            f.write('{"t": 17')       # torn by a power cut
        self.assertFalse(b.record(s1))
        self.assertTrue(b.record(s2))
        self.assertEqual([x['t'] for x in a.read_day(dt.date(2026, 11, 7))],
                         [self.NOW, self.NOW + 300])

    def test_total_deadline(self):
        def slow(url, timeout):
            time.sleep(3)
        with mock.patch.object(openmeteo, '_get_json_blocking', side_effect=slow):
            start = time.time()
            with self.assertRaises(RuntimeError):
                openmeteo.fetch(50.0, 14.9, timeout=0.3)
            self.assertLess(time.time() - start, 1.5)

    def test_broken_latest_is_none(self):
        store = openmeteo.OpenMeteoStore(self.base)
        self.assertIsNone(store.latest())
        os.makedirs(store.dir)
        with open(store.latest_path, 'w') as f:
            f.write('{"fetched_at": 1, "hourly": [{"time": "x"}]}')
        self.assertIsNone(store.latest())
        with open(store.latest_path, 'w') as f:
            f.write('{"fetched_at": 1, "hou')
        self.assertIsNone(store.latest())


class Priority(Tmp):
    NOW = Summary.NOW
    make = Summary.make

    def radar_hours(self, start, end, rainy=()):
        log = radar_log.RadarLog(self.base)
        for t in range(start, end + 1, 300):
            rn = t in rainy
            log.record(radar_log.sample_from(rain_data(t, rain_now=30 if rn else 0), rn, False))

    def rows(self, rain):
        h = 3600
        return [row(self.NOW + k * h, precip_mm=rain.get(k, 0.0)) for k in range(-48, 49)]

    def test_fresh_radar_overrides_model(self):
        # model: 1 mm now and 2 h ago, 1 mm 5 h ago; radar saw the last 3 h dry
        self.make(self.rows({0: 1.0, -2: 1.0, -5: 1.0}))
        self.radar_hours(self.NOW - 3 * 3600, self.NOW)
        s = history.summary(self.NOW, base=self.base)
        self.assertFalse(s['rain_last_h'])
        self.assertEqual(s['rain_last_h_mm'], 1.0)
        self.assertEqual(s['last_rain_source'], 'open-meteo')
        self.assertEqual(s['hours_since_rain'], 5.0)  # model fills the hours radar did not see

    def test_radar_rain_beats_model(self):
        self.make(self.rows({-1: 1.0}))
        self.radar_hours(self.NOW - 3 * 3600, self.NOW, rainy={self.NOW - 2 * 3600})
        s = history.summary(self.NOW, base=self.base)
        self.assertEqual((s['last_rain_source'], s['hours_since_rain']), ('radar', 2.0))

    def test_stale_radar_falls_back_to_model(self):
        self.make(self.rows({0: 1.0}), fetched=self.NOW + 3000)
        self.radar_hours(self.NOW - 3 * 3600, self.NOW - 3600)  # ends 1 h ago
        s = history.summary(self.NOW + 60, base=self.base)
        self.assertTrue(s['stale']['radar'])
        self.assertTrue(s['rain_last_h'])
        self.assertEqual(s['last_rain_source'], 'open-meteo')

    def test_drizzle_is_not_rain(self):
        self.make(self.rows({-1: 0.1, 3: 0.1}))
        s = history.summary(self.NOW, base=self.base)
        self.assertIsNone(s['hours_since_rain'])
        self.assertEqual(s['forecast_window_dry_h'], 48.0)

    def test_stale_forecast_is_none(self):
        self.make(self.rows({3: 5.0}), fetched=self.NOW - 4 * 3600)
        s = history.summary(self.NOW, base=self.base)
        self.assertTrue(s['stale']['forecast'])
        for key in ('forecast_rain_next_h', 'forecast_prob_next_h', 'forecast_rain_next_6h_mm',
                    'forecast_rain_next_24h_mm', 'forecast_window_dry_h',
                    'temp_min_next_24h', 'temp_max_next_24h'):
            self.assertIsNone(s[key], key)
        self.assertFalse(s['forecast_window_capped'])
        # a fresh radar alert still closes the window
        radar_log.RadarLog(self.base).record(
            radar_log.sample_from(rain_data(self.NOW - 60), False, True))
        self.assertEqual(history.summary(self.NOW, base=self.base)['forecast_window_dry_h'], 0.0)

if __name__ == '__main__':
    unittest.main()
