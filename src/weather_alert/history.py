"""Weather history + forecast in one dict. No rospy.

``summary(now)`` is the one input for the agent's mowing decisions and for the
planner's hard limits ("3 h after rain", "below 5 °C"). It reads only files
under ``~/.vitulus/weather`` (written by the weather_alert and
weather_openmeteo nodes), so it works with ROS down and without network.

    >>> from weather_alert import history
    >>> s = history.summary()
    >>> s['hours_since_rain'], s['temp_now'], s['forecast_window_dry_h']

Every value can be None when the data behind it is missing; ``stale`` and
``*_age_s`` say how old the data is.
"""
import datetime as dt
import time

from weather_alert import storage
from weather_alert.openmeteo import OpenMeteoStore, RAIN_MM
from weather_alert.radar_log import RadarLog

RAIN_PROB_PCT = 50       # a forecast hour this likely to rain is not dry
FORECAST_STALE_S = 3 * 3600
RADAR_STALE_S = 30 * 60
RADAR_COVER_FRAMES = 9   # of 12 five-minute frames: radar "saw" that hour
HISTORY_LOOKBACK_DAYS = 30
FORECAST_HORIZON_H = 48


def _epoch(now):
    if now is None:
        return time.time()
    if isinstance(now, dt.datetime):
        return now.timestamp()
    return float(now)


def _sum(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals), 2) if vals else None


def _hours(seconds):
    return None if seconds is None else round(seconds / 3600.0, 2)


def summary(now=None, base=None, rain_mm=RAIN_MM, rain_prob_pct=RAIN_PROB_PCT):
    now = _epoch(now)
    om = OpenMeteoStore(base)
    radar = RadarLog(base)
    latest = om.latest()
    status = om.status()

    # ---- Open-Meteo: history (archive + latest) and forecast ----
    past = om.hourly(now - 7 * 86400, now)
    past_24 = [r for r in past if r['time'] > now - 86400]
    last_hour = [r for r in past if r['time'] > now - 3600]
    # the row at T covers (T-1h, T]: the hour in progress is the next row
    future = [r for r in (latest or {}).get('hourly') or []
              if now < r['time'] <= now + FORECAST_HORIZON_H * 3600]
    forecast_age = int(now - latest['fetched_at']) if latest else None
    forecast_stale = forecast_age is None or forecast_age > FORECAST_STALE_S
    # an old forecast must not look valid: every forecast_* is None then
    forecast = [] if forecast_stale else future

    def om_rainy_rows():
        """Rainy model hours, newest first, up to 30 days back."""
        for r in reversed(past):
            if (r.get('precip_mm') or 0) >= rain_mm:
                yield r
        older = om.hourly(now - HISTORY_LOOKBACK_DAYS * 86400, now - 7 * 86400)
        for r in reversed(older):
            if (r.get('precip_mm') or 0) >= rain_mm:
                yield r

    # ---- radar ----
    samples = radar.samples(now - 86400, now)
    last_sample = samples[-1] if samples else None
    radar_age = int(now - last_sample['t']) if last_sample else None
    radar_fresh = radar_age is not None and radar_age <= RADAR_STALE_S
    radar_1h = [s for s in samples if s['t'] > now - 3600]
    radar_last_rain = radar.last_rain(now, days=HISTORY_LOOKBACK_DAYS)

    def rain_minutes(items):
        minutes = 0.0
        for i, s in enumerate(items):
            if not s.get('rain'):
                continue
            nxt = items[i + 1]['t'] if i + 1 < len(items) else s['t'] + 300
            minutes += min(nxt - s['t'], 600) / 60.0
        return round(minutes, 1)

    def radar_covers(until):
        """Radar has (nearly) every frame of the hour (until-1h, until]."""
        return len(radar.samples(until - 3600, until)) >= RADAR_COVER_FRAMES

    # ---- combined last rain ----
    # Fresh radar is the observation: where it has the frames, it decides and
    # the model only fills the hours radar did not see. Stale radar: the
    # newer of both, as before.
    om_last_rain = None
    for r in om_rainy_rows():
        if radar_fresh and radar_covers(r['time']):
            continue  # radar saw this hour (and its rain, if any)
        om_last_rain = r['time']
        break
    candidates = [(t, src) for t, src in ((radar_last_rain, 'radar'),
                                          (om_last_rain, 'open-meteo')) if t]
    last_rain, last_rain_src = max(candidates) if candidates else (None, None)
    hours_since_rain = _hours(now - last_rain) if last_rain else None
    if hours_since_rain is not None:
        hours_since_rain = max(0.0, hours_since_rain)

    rain_last_h_mm = _sum(last_hour, 'precip_mm')
    radar_rain_1h = any(s.get('rain') for s in radar_1h)
    if radar_fresh and len(radar_1h) >= RADAR_COVER_FRAMES:
        rain_last_h = bool(radar_rain_1h)
    else:
        rain_last_h = bool(radar_rain_1h or (rain_last_h_mm or 0) >= rain_mm)

    # ---- now ----
    cur = (latest or {}).get('current') or {}
    nearest = None
    rows_all = past + future
    if rows_all:
        nearest = min(rows_all, key=lambda r: abs(r['time'] - now))
        if abs(nearest['time'] - now) > 2 * 3600:
            nearest = None
    use_cur = cur and cur.get('time') and abs(cur['time'] - now) <= 3600

    def now_value(key):
        if use_cur and cur.get(key) is not None:
            return cur[key]
        return nearest.get(key) if nearest else None

    # ---- forecast ----
    next_h = forecast[:1]
    dry_h = None
    radar_alert = bool(radar_fresh and (last_sample.get('alert') or last_sample.get('rain')))
    if radar_alert:
        dry_h = 0.0
    elif forecast:
        dry_h = _hours(forecast[-1]['time'] - now)
        for r in forecast:
            if ((r.get('precip_mm') or 0) >= rain_mm
                    or (r.get('precip_prob_pct') or 0) >= rain_prob_pct):
                dry_h = max(0.0, _hours(r['time'] - 3600 - now))
                break
    next_24 = [r for r in forecast if r['time'] <= now + 86400]
    temps_24 = [r['temp_c'] for r in next_24 if r.get('temp_c') is not None]

    rain_7d = _sum(past, 'precip_mm')
    et0_7d = _sum(past, 'et0_mm')
    return {
        'now': int(now),
        'now_iso': storage.iso(now),
        # history
        'rain_last_h': rain_last_h,
        'rain_last_h_mm': rain_last_h_mm,
        'radar_rain_min_1h': rain_minutes(radar_1h),
        'radar_rain_min_24h': rain_minutes(samples),
        'rain_24h_mm': _sum(past_24, 'precip_mm'),
        'rain_7d_mm': rain_7d,
        'hours_since_rain': hours_since_rain,
        'last_rain': last_rain,
        'last_rain_iso': storage.iso(last_rain),
        'last_rain_source': last_rain_src,
        'no_rain_within_days': None if last_rain else HISTORY_LOOKBACK_DAYS,
        'et0_24h_mm': _sum(past_24, 'et0_mm'),
        'et0_7d_mm': et0_7d,
        'water_balance_7d_mm': (round(rain_7d - et0_7d, 2)
                                if rain_7d is not None and et0_7d is not None else None),
        # now
        'temp_now': now_value('temp_c'),
        'humidity_now': now_value('rh_pct'),
        'wind_now_kmh': now_value('wind_kmh'),
        'gust_now_kmh': nearest.get('gust_kmh') if nearest else None,
        'soil_moist_now': nearest.get('soil_moist_m3m3') if nearest else None,
        # forecast
        'forecast_rain_next_h': _sum(next_h, 'precip_mm'),
        'forecast_prob_next_h': next_h[0].get('precip_prob_pct') if next_h else None,
        'forecast_rain_next_6h_mm': _sum([r for r in forecast if r['time'] <= now + 6 * 3600],
                                         'precip_mm') if forecast else None,
        'forecast_rain_next_24h_mm': _sum(next_24, 'precip_mm') if forecast else None,
        'forecast_window_dry_h': dry_h,
        'forecast_window_capped': bool(forecast and not radar_alert and dry_h is not None
                                       and dry_h >= _hours(forecast[-1]['time'] - now)),
        'temp_min_next_24h': min(temps_24) if temps_24 else None,
        'temp_max_next_24h': max(temps_24) if temps_24 else None,
        # radar now
        'radar': {
            'status_now': last_sample.get('st') if last_sample else None,
            'rain_now': bool(last_sample.get('rain')) if last_sample else None,
            'rain_alert': bool(last_sample.get('alert')) if last_sample else None,
            'coverage_now_pct': (last_sample.get('now') or [None])[0] if last_sample else None,
            'frame_time': last_sample['t'] if last_sample else None,
        },
        # freshness
        'forecast_age_s': forecast_age,
        'radar_age_s': radar_age,
        'stale': {
            'forecast': forecast_stale,
            'radar': not radar_fresh,
        },
        'forecast_last_error': status.get('last_error'),
        'location': {'lat': (latest or {}).get('lat'), 'lon': (latest or {}).get('lon'),
                     'site': (latest or {}).get('site')},
    }


def history(now=None, base=None, days=HISTORY_LOOKBACK_DAYS):
    """The /weather/history document: per day Open-Meteo + radar summaries."""
    now = _epoch(now)
    today = storage.local_date(now)
    om_days = {d['date']: d for d in OpenMeteoStore(base).history_days(today + dt.timedelta(days=1), days + 1)}
    radar_days = RadarLog(base).daily()
    out = []
    for back in range(days, -1, -1):
        key = (today - dt.timedelta(days=back)).isoformat()
        if key in om_days or key in radar_days:
            out.append({'date': key, 'openmeteo': om_days.get(key),
                        'radar': radar_days.get(key)})
    return {'now': int(now), 'now_iso': storage.iso(now), 'days': out}
