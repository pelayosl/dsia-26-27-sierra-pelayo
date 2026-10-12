# ENERGY FORECAST APP

Parte B del Proyecto I de DSIA.

---
## Datos

### Dataset completo

Demanda eléctrica horaria de España combinada con meteorología, generada con el proyecto
[`dsia-dataset`](https://github.com/pelayosl/dsia-dataset) (`python -m src.build_dataset --start 2019-01-01`).

| Métrica | Valor |
|---|---|
| Fichero | `energy_weather_es_hourly.csv` |
| Periodo | `2019-01-01T00:00:00+00:00` → `2026-10-11T19:00:00+00:00` |
| Filas | 68,180 (una por hora, sin huecos en la rejilla) |
| Columnas | 12 |
| Tamaño | 7.1 MB |
| Fecha de generación | 2026-10-11 |

El dataset completo (~68.000 filas) **no se incluye en el repositorio** por tamaño (el enunciado pide una muestra de miles de filas como máximo).

### Muestra incluida en el repositorio

| Métrica | Valor |
|---|---|
| Fichero | [`data/energy_weather_es_hourly_sample.csv`](../data/energy_weather_es_hourly_sample.csv) |
| Periodo | `2025-04-01T00:00:00+00:00` → `2026-03-31T23:00:00+00:00` (UTC) |
| Filas | 8,760 (12,8 % del dataset completo) |
| Tamaño | 0.9 MB |

La muestra es un recorte contiguo del dataset completo, con las mismas columnas y formato. Se ha elegido este año porque contiene:

- Un ciclo estacional completo (invierno y verano), necesario para relacionar demanda y temperatura.
- Los dos cambios de hora (2025-10-26 con 25 h locales y 2026-03-29 con 23 h).
- 9 días festivos nacionales.
- Casos inválidos reales: 35 horas sin demanda (apagón del 28-29 de abril de 2025) y 2 valores de demanda fuera de rango (2026-03-17).

No incluye el periodo COVID (`covid_flag` es siempre 0 en la muestra).

### Columnas

| Columna | Tipo | Descripción |
|---|---|---|
| `time_utc` | datetime | Inicio de la hora, en UTC |
| `time_local` | datetime | Inicio de la hora, en `Europe/Madrid` (offsets mixtos +01:00 / +02:00) |
| `load_mw` | float | Demanda media horaria (MW) |
| `temperature_2m` | float | °C, media ponderada por población de 5 ciudades |
| `relative_humidity_2m` | float | %, media ponderada |
| `wind_speed_10m` | float | km/h, media ponderada |
| `shortwave_radiation` | float | W/m², media ponderada |
| `precipitation` | float | mm, media ponderada |
| `is_holiday` | int (0/1) | Festivo nacional en España |
| `covid_flag` | int (0/1) | Estado de alarma (2020-03-14 a 2020-06-21) |
| `load_interpolated` | int (0/1) | Demanda interpolada |
| `weather_interpolated` | int (0/1) | Alguna variable meteorológica interpolada |

### Fuentes y licencias

- **Demanda:** [Energy-Charts](https://energy-charts.info) (Fraunhofer ISE), con datos de la
  [ENTSO-E Transparency Platform](https://transparency.entsoe.eu). Licencia
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); los datos de origen ENTSO-E están sujetos a sus condiciones de uso.
  Cita: *Energy-Charts.info, Fraunhofer ISE, datos de ENTSO-E*.
- **Meteorología:** [Open-Meteo](https://open-meteo.com). Licencia [CC BY 4.0](https://open-meteo.com/en/license);
  la API gratuita es solo para uso no comercial. Cita: *Weather data by Open-Meteo.com*.
