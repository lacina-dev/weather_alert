"""Permanent record of the ČHMÚ radar status (what /weather_alert/rain_alert says).

One compact JSON line per radar frame into ``radar/YYYY-MM-DD.jsonl``::

    {"t": 1794095100,            radar NOW frame time (epoch)
     "rx": 1794095530,           when the robot stored it
     "st": "OK",                 final status of the NOW frame
     "rain": 0, "alert": 0,      the RainAlert flags
     "now": [r, a, w],           rain coverage % in the 1 / 9 / 20 km frames
     "past": [[-10, r, a, w], ...],   -10 .. -60 min
     "nc":   [[10, r, a, w], ...]}    nowcast +10 .. +30 min

The radar gives coverage of the frames around the robot, not mm/h; "intensity"
in the summaries is that coverage (rain frame %).

``radar/daily.json`` keeps a summary per day: rain minutes, max coverage, last
rain time. No rospy.
"""
import datetime as dt
import json
import os
import threading
import time

from weather_alert import storage

PAST_OFFSETS = (-10, -20, -30, -40, -50, -60)
NOWCAST_OFFSETS = (10, 20, 30)
#: one ČHMÚ frame stands for 5 minutes; a gap longer than this is "no data"
FRAME_MIN = 5.0
MAX_GAP_MIN = 10.0

#: T22 (N8): parsed day files, path -> (mtime_ns, size, samples). summary()
#: reads up to 30 days of frames, most of them several times per call; a
#: day file only grows, so a file with the same mtime and size is the same.
#: Measured on a 30-day copy: 260 ms -> ~10 ms per summary() (notebook).
_DAY_CACHE = {}
_DAY_CACHE_MAX = 64
_DAY_LOCK = threading.Lock()


def _r1(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _triple(obs):
    return [_r1(obs.get('percent_rain')), _r1(obs.get('percent_alert')),
            _r1(obs.get('percent_warn'))]


def _pick(observations, target, tolerance):
    if not observations:
        return None
    best = min(observations, key=lambda o: abs(int(o['time']) - target))
    if abs(int(best['time']) - target) > tolerance:
        return None
    return best


def sample_from(rain_data, rain_now, rain_alert, received=None):
    """Compact sample from ``radar.evaluate_data()``'s rain_data and the flags
    the node published. None when there is no NOW frame."""
    observations = rain_data.get('observations') or []
    past = [o for o in observations if o.get('type') != 'NOWCAST']
    nowcasts = [o for o in observations if o.get('type') == 'NOWCAST']
    now_obs = next((o for o in reversed(past) if o.get('type') == 'NOW'), None)
    if now_obs is None and past:
        now_obs = past[-1]
    if now_obs is None:
        return None
    ref = int(now_obs['time'])
    sample = {
        't': ref,
        'rx': int(received if received is not None else time.time()),
        'st': str(now_obs.get('final_status', '')),
        'rain': 1 if rain_now else 0,
        'alert': 1 if rain_alert else 0,
        'now': _triple(now_obs),
        'past': [],
        'nc': [],
    }
    for off in PAST_OFFSETS:
        obs = _pick(past, ref + off * 60, 300)
        if obs is not None:
            sample['past'].append([off] + _triple(obs))
    for off in NOWCAST_OFFSETS:
        obs = _pick(nowcasts, ref + off * 60, 600)
        if obs is not None:
            sample['nc'].append([off] + _triple(obs))
    return sample


def day_summary(samples):
    """Summary of one day's samples (sorted by t)."""
    samples = sorted(samples, key=lambda s: s['t'])
    rain_min = 0.0
    covered_min = 0.0
    max_rain = None
    max_alert = None
    last_rain = None
    first_rain = None
    for i, s in enumerate(samples):
        if i + 1 < len(samples):
            span = min((samples[i + 1]['t'] - s['t']) / 60.0, MAX_GAP_MIN)
        else:
            span = FRAME_MIN
        covered_min += span
        now = s.get('now') or [None, None, None]
        if now[0] is not None:
            max_rain = now[0] if max_rain is None else max(max_rain, now[0])
        if now[1] is not None:
            max_alert = now[1] if max_alert is None else max(max_alert, now[1])
        if s.get('rain'):
            rain_min += span
            last_rain = s['t']
            if first_rain is None:
                first_rain = s['t']
    return {
        'samples': len(samples),
        'rain_minutes': round(rain_min, 1),
        'covered_minutes': round(covered_min, 1),
        'max_rain_pct': max_rain,
        'max_alert_pct': max_alert,
        'first_rain': first_rain,
        'first_rain_iso': storage.iso(first_rain),
        'last_rain': last_rain,
        'last_rain_iso': storage.iso(last_rain),
        'first_t': samples[0]['t'] if samples else None,
        'last_t': samples[-1]['t'] if samples else None,
    }


class RadarLog(object):
    def __init__(self, base=None):
        self.dir = os.path.join(storage.base_dir(base), 'radar')

    def day_path(self, day):
        return os.path.join(self.dir, '%s.jsonl' % day.isoformat())

    @property
    def daily_path(self):
        return os.path.join(self.dir, 'daily.json')

    def read_day(self, day):
        """Samples of one day file (a list of its own; the samples must not
        be changed by the caller, they are shared with the cache)."""
        path = self.day_path(day)
        try:
            st = os.stat(path)
        except OSError:
            return []
        key = (st.st_mtime_ns, st.st_size)
        with _DAY_LOCK:
            hit = _DAY_CACHE.get(path)
        if hit is not None and hit[0] == key:
            return list(hit[1])
        out = self._parse_day(path)
        with _DAY_LOCK:
            if path not in _DAY_CACHE and len(_DAY_CACHE) >= _DAY_CACHE_MAX:
                _DAY_CACHE.pop(next(iter(_DAY_CACHE)))
            _DAY_CACHE[path] = (key, out)
        return list(out)

    @staticmethod
    def _parse_day(path):
        out = []
        try:
            with open(path) as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue  # a line torn by a power cut
                    if isinstance(item, dict) and isinstance(item.get('t'), int):
                        out.append(item)
        except (OSError, UnicodeDecodeError):
            pass
        return out

    def last_t(self, day):
        """t of the last stored frame of ``day``, read from the FILE (so a
        second writer or a restart cannot store a frame twice)."""
        try:
            return int(json.loads(storage.last_line(self.day_path(day)))['t'])
        except (ValueError, KeyError, TypeError):
            items = self.read_day(day)  # last line torn: scan the day
            return max((i['t'] for i in items), default=0)

    def record(self, sample):
        """Append a sample; a frame already stored is skipped. True if written."""
        if not sample:
            return False
        day = storage.local_date(sample['t'])
        if sample['t'] <= self.last_t(day):
            return False
        storage.append_line(self.day_path(day),
                            json.dumps(sample, sort_keys=True, separators=(',', ':')))
        daily = storage.read_json(self.daily_path, {}) or {}
        daily[day.isoformat()] = day_summary(self.read_day(day))
        storage.write_json(self.daily_path, daily)
        return True

    def daily(self):
        return storage.read_json(self.daily_path, {}) or {}

    def samples(self, since, until):
        """Samples with since < t <= until (epoch), sorted."""
        day = storage.local_date(since)
        end = storage.local_date(until)
        out = []
        while day <= end:
            out.extend(s for s in self.read_day(day) if since < s['t'] <= until)
            day += dt.timedelta(days=1)
        return sorted(out, key=lambda s: s['t'])

    def last_rain(self, until, days=30):
        """Epoch of the last rainy frame at or before ``until`` within ``days``."""
        day = storage.local_date(until)
        for _ in range(days):
            rainy = [s['t'] for s in self.read_day(day) if s.get('rain') and s['t'] <= until]
            if rainy:
                return max(rainy)
            day -= dt.timedelta(days=1)
        return None
