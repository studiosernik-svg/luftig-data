# luftig-data

Ежедневный прогноз пыльцы для приложения Luftig: сетка по Германии из
Copernicus CAMS European air quality ensemble (0,1°, по часам на 96 ч;
ольха, берёза, злаки, полынь, амброзия).

Файл: `https://studiosernik-svg.github.io/luftig-data/pollen-de.bin.gz`
(формат — в начале `pipeline/cams/cams_pollen.py`), метаданные — `latest.json`.

Собирается GitHub Actions раз в сутки (`.github/workflows/cams-pollen.yml`).
Нужен секрет `ADS_KEY` — ключ Atmosphere Data Store.

Contains modified Copernicus Atmosphere Monitoring Service information.
Pollen forecasts from CAMS are experimental (not validated against observations).
Исходники конвейера ведутся в основном репозитории Luftig (`pipeline/cams/`).
