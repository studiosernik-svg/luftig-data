"""Конвейер пыльцы CAMS для Luftig.

Раз в сутки скачивает прогноз Copernicus CAMS European air quality (ансамбль,
0,1°, по часам на 96 ч) по пяти видам пыльцы, вырезает Германию и пишет один
компактный файл `pollen-de.bin.gz` + `latest.json`. Приложение скачивает этот
файл целиком — координаты пользователя никуда не уходят.

Формат `pollen-de.bin` (до gzip):
  строка JSON-заголовка в UTF-8, затем `\\n`, затем байты uint8
  в порядке [вид][час][широта с севера на юг][долгота с запада на восток].
  Значение q кодирует концентрацию c (Pollen/m³): q = round(20·log2(1 + c)),
  обрезано до 0…255; обратно c = 2^(q/20) − 1 (шаг около 3,5 %).

Данные: Copernicus Atmosphere Monitoring Service, лицензия Copernicus
(свободно, в т. ч. коммерчески, с указанием источника). Прогноз пыльцы CAMS
не валидирован по наблюдениям и считается экспериментальным.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

DATASET = 'cams-europe-air-quality-forecasts'
# Порядок фиксирован: приложение читает виды по заголовку, но тесты на него опираются.
SPECIES = ['alder', 'birch', 'grass', 'mugwort', 'ragweed']
# Германия с запасом: север, запад, юг, восток.
AREA = [55.2, 5.5, 47.0, 15.5]
HOURS = 97  # 0…96 ч от 00 UTC
SCALE = 20.0
FORMAT_VERSION = 1


def encode(c: np.ndarray) -> np.ndarray:
    """Концентрация (Pollen/m³) → uint8 по логарифмической шкале."""
    c = np.nan_to_num(np.asarray(c, dtype=np.float64), nan=0.0)
    q = np.rint(SCALE * np.log2(1.0 + np.clip(c, 0.0, None)))
    return np.clip(q, 0, 255).astype(np.uint8)


def decode(q: np.ndarray) -> np.ndarray:
    return np.power(2.0, np.asarray(q, dtype=np.float64) / SCALE) - 1.0


def request(date: dt.date) -> dict:
    d = date.isoformat()
    return {
        'variable': [f'{s}_pollen' for s in SPECIES],
        'model': ['ensemble'],
        'level': ['0'],
        'date': [f'{d}/{d}'],
        'type': ['forecast'],
        'time': ['00:00'],
        'leadtime_hour': [str(h) for h in range(HOURS)],
        'data_format': 'netcdf_zip',
        'area': AREA,
    }


def download(date: dt.date, target: Path) -> Path:
    import cdsapi  # только для настоящей загрузки; тестам не нужен

    cdsapi.Client().retrieve(DATASET, request(date), str(target))
    return target


def _species_var(ds: xr.Dataset, species: str) -> xr.DataArray:
    """Переменная вида: по имени или по описанию (в NetCDF CAMS имена короткие)."""
    for name, var in ds.data_vars.items():
        text = ' '.join(
            str(x).lower() for x in (name, var.attrs.get('long_name', ''), var.attrs.get('species', ''))
        )
        if species in text and 'pollen' in text or text.split()[0] == f'{species}_pollen':
            return var
    raise KeyError(f'no variable for {species}: {list(ds.data_vars)}')


def open_netcdf(path: Path) -> xr.Dataset:
    """NetCDF или ZIP с несколькими NetCDF (так отдаёт ADS при netcdf_zip)."""
    if zipfile.is_zipfile(path):
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        files = sorted(tmp.glob('*.nc'))
        return xr.merge([xr.open_dataset(f) for f in files], compat='override')
    return xr.open_dataset(path)


def build(ds: xr.Dataset, issued: dt.datetime) -> tuple[dict, bytes]:
    """Сетка Германии → (заголовок, байты uint8)."""
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    ds = ds.sortby(lat_name, ascending=False).sortby(lon_name)
    lon = ds[lon_name].values
    if lon.max() > 180:  # 0…360 → −180…180
        ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180}).sortby(lon_name)
    ds = ds.sel(
        {lat_name: slice(AREA[0], AREA[2]), lon_name: slice(AREA[1], AREA[3])}
    )
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    blocks = []
    for s in SPECIES:
        v = _species_var(ds, s)
        extra = [d for d in v.dims if d not in (lat_name, lon_name)]
        time_dim = next(d for d in extra if v.sizes[d] > 1) if any(v.sizes[d] > 1 for d in extra) else None
        for d in extra:
            if d != time_dim:
                v = v.isel({d: 0})
        arr = v.transpose(*( [time_dim] if time_dim else []), lat_name, lon_name).values
        if time_dim is None:
            arr = arr[np.newaxis]
        if arr.shape[0] < HOURS:  # короче 97 ч — дополняем последним значением
            arr = np.concatenate([arr, np.repeat(arr[-1:], HOURS - arr.shape[0], 0)])
        blocks.append(encode(arr[:HOURS]))
    data = np.stack(blocks).astype(np.uint8)
    header = {
        'v': FORMAT_VERSION,
        'issued': issued.strftime('%Y-%m-%dT%H:%MZ'),
        'species': SPECIES,
        'hours': HOURS,
        'lat0': round(float(lats[0]), 4),
        'lon0': round(float(lons[0]), 4),
        'dlat': round(float(lats[1] - lats[0]), 4),
        'dlon': round(float(lons[1] - lons[0]), 4),
        'nlat': int(len(lats)),
        'nlon': int(len(lons)),
        'scale': SCALE,
        'source': 'Copernicus Atmosphere Monitoring Service (CAMS), European air quality ensemble forecast',
    }
    return header, data.tobytes()


def write(header: dict, data: bytes, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(header, separators=(',', ':')).encode() + b'\n' + data
    gz = gzip.compress(raw, compresslevel=9, mtime=0)
    (out / 'pollen-de.bin.gz').write_bytes(gz)
    meta = {
        'issued': header['issued'],
        'file': 'pollen-de.bin.gz',
        'bytes': len(gz),
        'sha256': hashlib.sha256(gz).hexdigest(),
    }
    (out / 'latest.json').write_text(json.dumps(meta, indent=1))
    return meta


def read(path: Path) -> tuple[dict, np.ndarray]:
    raw = gzip.decompress(path.read_bytes())
    nl = raw.index(b'\n')
    header = json.loads(raw[:nl])
    shape = (len(header['species']), header['hours'], header['nlat'], header['nlon'])
    return header, np.frombuffer(raw[nl + 1:], dtype=np.uint8).reshape(shape)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--date', default=dt.date.today().isoformat(), help='дата прогона (UTC), по умолчанию сегодня')
    p.add_argument('--input', type=Path, help='готовый NetCDF/ZIP вместо загрузки')
    p.add_argument('--out', type=Path, default=Path('public'))
    a = p.parse_args(argv)
    date = dt.date.fromisoformat(a.date)
    src = a.input or download(date, Path(tempfile.mkdtemp()) / 'cams.zip')
    header, data = build(open_netcdf(src), dt.datetime.combine(date, dt.time(0), tzinfo=dt.timezone.utc))
    meta = write(header, data, a.out)
    print(json.dumps(meta))
    return 0


if __name__ == '__main__':
    sys.exit(main())
