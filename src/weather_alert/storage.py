"""Weather data on disk: paths, atomic JSON files, the site location.

No rospy here: the planner, the agent and tests import this directly.

Layout under ``~/.vitulus/weather`` (``VITULUS_WEATHER_DIR`` overrides it)::

    radar/YYYY-MM-DD.jsonl        one compact line per ČHMÚ radar frame
    radar/daily.json              per-day radar summary
    openmeteo/latest.json         last successful Open-Meteo answer (parsed)
    openmeteo/status.json         last attempt / last error
    openmeteo/archive/YYYY-MM-DD.json   hourly rows + daily summary of a past day
"""
import datetime as dt
import json
import os
import tempfile

import yaml

FILE_MODE = 0o644


def base_dir(base=None):
    if base:
        return base
    env = os.environ.get('VITULUS_WEATHER_DIR')
    if env:
        return env
    return os.path.join(os.path.expanduser('~'), '.vitulus', 'weather')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, data):
    """Write ``data`` atomically (tmp + fsync + rename): a power cut leaves
    either the old or the new file, never half of one."""
    ensure_dir(os.path.dirname(path))
    fd, tmp = tempfile.mkstemp(prefix='.tmp_', dir=os.path.dirname(path))
    try:
        os.fchmod(fd, FILE_MODE)  # mkstemp makes 0600; others read these too
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, sort_keys=True, separators=(',', ':'), default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_line(path, line):
    """Append one line (fsync); a new file is created 0644."""
    ensure_dir(os.path.dirname(path))
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, FILE_MODE)
    try:
        size = os.fstat(fd).st_size
        torn = size > 0 and os.pread(fd, 1, size - 1) != b'\n'
        os.write(fd, (('\n' if torn else '') + line + '\n').encode('utf-8'))
        os.fsync(fd)
    finally:
        os.close(fd)


def last_line(path, max_bytes=8192):
    """Last complete line of a file ('' when none)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ''
    lines = data.decode('utf-8', errors='replace').split('\n')
    complete = lines[:-1]  # the piece after the last newline is torn or ''
    return complete[-1] if complete else ''


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def local_date(ts):
    """Local calendar date (system TZ, Europe/Prague on the robot) of epoch ts."""
    return dt.datetime.fromtimestamp(float(ts)).date()


def day_bounds(day):
    """(start, end) epoch of a local calendar day."""
    start = dt.datetime.combine(day, dt.time())
    end = start + dt.timedelta(days=1)
    return start.timestamp(), end.timestamp()


def iso(ts):
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(float(ts)).isoformat(timespec='seconds')


def site_location(mapping_dir=None):
    """(lat, lon, site) of the served site's datum, or None.

    The active site is ``~/.vitulus/mapping_v3/.last_served``; its datum is
    ``<site>/datum.yaml`` (origin_lat/lon, robot_lat/lon as a fallback)."""
    root = mapping_dir or os.path.join(os.path.expanduser('~'), '.vitulus', 'mapping_v3')
    served = read_json(os.path.join(root, '.last_served'), {}) or {}
    site = served.get('site') if isinstance(served, dict) else None
    if not site:
        return None
    try:
        with open(os.path.join(root, site, 'datum.yaml')) as f:
            datum = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    for lat_key, lon_key in (('origin_lat', 'origin_lon'), ('robot_lat', 'robot_lon')):
        try:
            lat, lon = float(datum[lat_key]), float(datum[lon_key])
        except (KeyError, TypeError, ValueError):
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, site
    return None
