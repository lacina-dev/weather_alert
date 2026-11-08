"""Open-Meteo forecast and recent history for the robot's site. No rospy.

One request to the forecast API with ``past_days`` gives both the hourly
history of the last days (the same model data Open-Meteo's archive serves for
recent days, without the archive's 2-5 day lag) and the forecast. Past days
are kept in ``openmeteo/archive/YYYY-MM-DD.json`` forever, so the robot builds
its own long history even though one request reaches back at most 92 days.

Hourly fields (Open-Meteo semantics): sums (precipitation, et0, sunshine) are
over the PRECEDING hour, i.e. the row at time T covers (T-1h, T]; the rest is
the value at T.
"""
import datetime as dt
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from weather_alert import storage

API_URL = 'https://api.open-meteo.com/v1/forecast'
USER_AGENT = 'Vitulus-weather/1.0'

#: Open-Meteo name -> our key
HOURLY = (
    ('temperature_2m', 'temp_c'),
    ('relative_humidity_2m', 'rh_pct'),
    ('precipitation', 'precip_mm'),
    ('precipitation_probability', 'precip_prob_pct'),
    ('weather_code', 'weather_code'),
    ('cloud_cover', 'cloud_pct'),
    ('wind_speed_10m', 'wind_kmh'),
    ('wind_gusts_10m', 'gust_kmh'),
    ('et0_fao_evapotranspiration', 'et0_mm'),
    ('soil_moisture_0_to_1cm', 'soil_moist_m3m3'),
    ('sunshine_duration', 'sun_s'),
)
CURRENT = (
    ('temperature_2m', 'temp_c'),
    ('relative_humidity_2m', 'rh_pct'),
    ('precipitation', 'precip_mm'),
    ('weather_code', 'weather_code'),
    ('wind_speed_10m', 'wind_kmh'),
)
RAIN_MM = 0.2  # a model hour with less than this is not rain (drizzle noise)


def build_params(lat, lon, past_days, forecast_days):
    return {
        'latitude': '%.5f' % lat, 'longitude': '%.5f' % lon,
        'timezone': 'Europe/Prague', 'timeformat': 'unixtime',
        'past_days': int(past_days), 'forecast_days': int(forecast_days),
        'hourly': ','.join(k for k, _ in HOURLY),
        'current': ','.join(k for k, _ in CURRENT),
        'wind_speed_unit': 'kmh', 'precipitation_unit': 'mm',
    }


def fetch(lat, lon, past_days=2, forecast_days=3, timeout=20.0):
    """Raw JSON payload. Raises RuntimeError on any network/API problem."""
    url = API_URL + '?' + urllib.parse.urlencode(build_params(lat, lon, past_days, forecast_days))
    payload = _get_json(url, timeout)
    if not isinstance(payload, dict) or payload.get('error'):
        raise RuntimeError('Open-Meteo error: %s' % (payload or {}).get('reason'))
    if not isinstance(payload.get('hourly'), dict) or not payload['hourly'].get('time'):
        raise RuntimeError('Open-Meteo returned no hourly data')
    return payload


def _get_json(url, timeout):
    """GET + JSON with ONE deadline for everything (DNS, connect, read).

    urlopen's timeout is per socket operation and does not cover the DNS
    lookup, so the request runs in a daemon thread and we wait at most
    ``timeout``; a hung lookup is left behind and dies with the process."""
    box = {}

    def work():
        try:
            box['ok'] = _get_json_blocking(url, timeout)
        except BaseException as exc:  # handed over to the caller
            box['err'] = exc

    worker = threading.Thread(target=work, name='openmeteo-get', daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise RuntimeError('Open-Meteo: no answer within %.0f s' % timeout)
    if 'err' in box:
        raise box['err']
    return box['ok']


def _get_json_blocking(url, timeout):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read().decode('utf-8')).get('reason')
        except Exception:
            reason = None
        raise RuntimeError('Open-Meteo HTTP %s: %s' % (exc.code, reason or exc.reason))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise RuntimeError('Open-Meteo unreachable: %s' % exc)


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def parse_hourly(payload):
    """Hourly rows [{'time': epoch, 'temp_c': ..., ...}] sorted by time."""
    hourly = payload.get('hourly') or {}
    times = hourly.get('time') or []
    rows = []
    for i, stamp in enumerate(times):
        if not isinstance(stamp, int):
            continue  # we always ask for timeformat=unixtime
        row = {'time': stamp}
        for src, key in HOURLY:
            values = hourly.get(src) or []
            row[key] = _num(values[i]) if i < len(values) else None
        rows.append(row)
    return sorted(rows, key=lambda r: r['time'])


def parse_current(payload):
    cur = payload.get('current') or {}
    if not isinstance(cur.get('time'), int):
        return None
    out = {'time': cur['time']}
    for src, key in CURRENT:
        out[key] = _num(cur.get(src))
    return out


def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def _sum(rows, key, digits=2):
    v = _vals(rows, key)
    return round(sum(v), digits) if v else None


def _mean(rows, key, digits=1):
    v = _vals(rows, key)
    return round(sum(v) / len(v), digits) if v else None


def day_summary(rows):
    """Daily aggregate of one local day's hourly rows."""
    temps = _vals(rows, 'temp_c')
    precip = _vals(rows, 'precip_mm')
    winds = _vals(rows, 'wind_kmh')
    gusts = _vals(rows, 'gust_kmh')
    sun = _vals(rows, 'sun_s')
    return {
        'hours': len(rows),
        'temp_min_c': min(temps) if temps else None,
        'temp_max_c': max(temps) if temps else None,
        'temp_mean_c': _mean(rows, 'temp_c'),
        'precip_mm': _sum(rows, 'precip_mm', 1),
        'precip_hours': sum(1 for p in precip if p >= RAIN_MM),
        'precip_prob_max_pct': max(_vals(rows, 'precip_prob_pct'), default=None),
        'et0_mm': _sum(rows, 'et0_mm', 2),
        'rh_mean_pct': _mean(rows, 'rh_pct', 0),
        'wind_max_kmh': max(winds) if winds else None,
        'gust_max_kmh': max(gusts) if gusts else None,
        'sun_h': round(sum(sun) / 3600.0, 1) if sun else None,
        'soil_moist_mean': _mean(rows, 'soil_moist_m3m3', 3),
    }


def group_days(rows):
    """{date: [rows]} by the local date of the row time (as Open-Meteo does)."""
    out = {}
    for row in rows:
        out.setdefault(storage.local_date(row['time']), []).append(row)
    return out


def daily_list(rows):
    return [dict(day_summary(day_rows), date=day.isoformat())
            for day, day_rows in sorted(group_days(rows).items())]


class OpenMeteoStore(object):
    """Files under ``<base>/openmeteo``."""

    def __init__(self, base=None):
        self.dir = os.path.join(storage.base_dir(base), 'openmeteo')

    @property
    def latest_path(self):
        return os.path.join(self.dir, 'latest.json')

    @property
    def status_path(self):
        return os.path.join(self.dir, 'status.json')

    def archive_path(self, day):
        return os.path.join(self.dir, 'archive', '%s.json' % day.isoformat())

    # -- writing ------------------------------------------------------------
    def save(self, payload, fetched_at, lat, lon, site=None):
        """Store a fetched payload: latest.json + the past part of every day
        into the archive. Returns the latest document."""
        rows = parse_hourly(payload)
        latest = {
            'fetched_at': int(fetched_at),
            'fetched_iso': storage.iso(fetched_at),
            'lat': lat, 'lon': lon, 'site': site,
            'elevation': payload.get('elevation'),
            'current': parse_current(payload),
            'hourly': rows,
        }
        storage.write_json(self.latest_path, latest)
        today = storage.local_date(fetched_at)
        for day, day_rows in group_days(rows).items():
            past = [r for r in day_rows if r['time'] <= fetched_at]
            if not past:
                continue
            _, end = storage.day_bounds(day)
            old = storage.read_json(self.archive_path(day)) or {}
            if old.get('final') and day < today - dt.timedelta(days=1):
                continue  # settled long ago, keep what we have
            storage.write_json(self.archive_path(day), {
                'date': day.isoformat(),
                'final': end <= fetched_at,
                'updated_at': int(fetched_at),
                'lat': lat, 'lon': lon, 'site': site,
                'daily': day_summary(past),
                'hourly': past,
            })
        self.set_status(ok=True, at=fetched_at)
        return latest

    def set_status(self, ok, at, error=None):
        status = storage.read_json(self.status_path, {}) or {}
        status['last_attempt'] = int(at)
        status['last_ok'] = bool(ok)
        if ok:
            status['last_success'] = int(at)
            status['last_error'] = None
        else:
            status['last_error'] = str(error)
            status['last_error_at'] = int(at)
        storage.write_json(self.status_path, status)
        return status

    # -- reading ------------------------------------------------------------
    def latest(self):
        """The last stored answer, or None when missing or not usable."""
        doc = storage.read_json(self.latest_path)
        if (not isinstance(doc, dict) or not isinstance(doc.get('fetched_at'), int)
                or not isinstance(doc.get('hourly'), list)
                or not all(isinstance(r, dict) and isinstance(r.get('time'), int)
                           for r in doc['hourly'])):
            return None
        return doc

    def status(self):
        return storage.read_json(self.status_path, {}) or {}

    def archive_day(self, day):
        return storage.read_json(self.archive_path(day))

    def missing_days(self, today, days):
        """Past days (yesterday back ``days``) not archived as final."""
        out = []
        for back in range(1, days + 1):
            day = today - dt.timedelta(days=back)
            doc = self.archive_day(day)
            if not doc or not doc.get('final'):
                out.append(day)
        return out

    def hourly(self, since, until):
        """Rows with since < time <= until: archive, overridden by latest."""
        by_time = {}
        day = storage.local_date(since)
        while day <= storage.local_date(until):
            for row in (self.archive_day(day) or {}).get('hourly') or []:
                by_time[row['time']] = row
            day += dt.timedelta(days=1)
        for row in (self.latest() or {}).get('hourly') or []:
            by_time[row['time']] = row
        return [by_time[t] for t in sorted(by_time) if since < t <= until]

    def history_days(self, today, days):
        """Daily summaries of the last ``days`` days before ``today`` (oldest first)."""
        out = []
        for back in range(days, 0, -1):
            day = today - dt.timedelta(days=back)
            doc = self.archive_day(day)
            if doc:
                out.append(dict(doc.get('daily') or {}, date=doc['date'],
                                final=bool(doc.get('final'))))
        return out


def forecast_view(latest, now, hours=48, stale_after_s=3 * 3600, status=None):
    """The /weather/forecast document: next ``hours`` hourly rows + days."""
    status = status or {}
    if not latest:
        return {'ok': False, 'now': int(now), 'age_s': None, 'stale': True,
                'hourly': [], 'daily': [], 'last_error': status.get('last_error'),
                'source': 'open-meteo'}
    rows = [r for r in latest.get('hourly') or [] if now < r['time'] <= now + hours * 3600]
    age = int(now - latest['fetched_at'])
    return {
        'ok': True,
        'source': 'open-meteo',
        'now': int(now),
        'fetched_at': latest['fetched_at'],
        'fetched_iso': latest.get('fetched_iso'),
        'age_s': age,
        'stale': age > stale_after_s,
        'last_error': status.get('last_error'),
        'lat': latest.get('lat'), 'lon': latest.get('lon'), 'site': latest.get('site'),
        'current': latest.get('current'),
        'hours': hours,
        'hourly': rows,
        'daily': daily_list(rows),
    }
