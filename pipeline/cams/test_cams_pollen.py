import datetime as dt

import numpy as np
import xarray as xr

import cams_pollen as cp


def fake_dataset(lon360=False):
    lat = np.round(np.arange(55.95, 46.0, -0.1), 2)  # шире области, с севера на юг
    lon = np.round(np.arange(4.05, 16.95, 0.1), 2)
    if lon360:
        lon = lon % 360
    t = np.arange(97)
    rng = np.random.default_rng(1)
    vars_ = {}
    short = {'alder': 'apg_conc', 'birch': 'bpg_conc', 'grass': 'gpg_conc', 'mugwort': 'mpg_conc', 'ragweed': 'rwpg_conc'}
    for i, s in enumerate(cp.SPECIES):
        base = np.fromfunction(lambda h, y, x: (x / 10.0) * (i + 1) + h * 0.5, (97, len(lat), len(lon)))
        vars_[short[s]] = (('time', 'level', 'latitude', 'longitude'), (base + rng.random(base.shape))[:, None],
                           {'long_name': f'{s.capitalize()} pollen', 'units': 'grains m**-3'})
    return xr.Dataset(vars_, coords={'time': t, 'level': [0], 'latitude': lat, 'longitude': lon})


def test_encode_roundtrip_within_4_percent():
    c = np.array([0, 0.5, 1, 9, 10, 11, 49, 50, 51, 140, 600, 2000.0])
    back = cp.decode(cp.encode(c))
    assert back[0] == 0
    assert np.all(np.abs(back[1:] - c[1:]) <= 0.04 * c[1:] + 0.05)
    assert cp.encode(np.array([1e9]))[0] == 255
    assert cp.encode(np.array([np.nan]))[0] == 0


def test_build_crops_germany_and_orders_species(tmp_path):
    for lon360 in (False, True):
        h, data = cp.build(fake_dataset(lon360), dt.datetime(2026, 4, 17, tzinfo=dt.timezone.utc))
        assert h['species'] == cp.SPECIES and h['hours'] == 97
        assert 55.0 <= h['lat0'] <= 55.2 and h['dlat'] < 0 and h['dlon'] > 0
        assert 5.5 <= h['lon0'] <= 5.6
        assert h['nlat'] == 82 and h['nlon'] == 100
        assert len(data) == 5 * 97 * 82 * 100
        meta = cp.write(h, data, tmp_path)
        h2, arr = cp.read(tmp_path / meta['file'])
        assert h2 == h and arr.shape == (5, 97, 82, 100)
        # Берёза (индекс 1) вдвое «гуще» ольхи на востоке в час 0.
        a, b = cp.decode(arr[0, 0, 40, -1]), cp.decode(arr[1, 0, 40, -1])
        assert 1.7 < b / a < 2.3
    assert meta['bytes'] < 2_000_000


def test_request_is_well_formed():
    r = cp.request(dt.date(2026, 10, 15))
    assert r['variable'] == ['alder_pollen', 'birch_pollen', 'grass_pollen', 'mugwort_pollen', 'ragweed_pollen']
    assert r['leadtime_hour'][0] == '0' and r['leadtime_hour'][-1] == '96'
    assert r['area'] == [55.2, 5.5, 47.0, 15.5] and r['date'] == ['2026-10-15/2026-10-15']
