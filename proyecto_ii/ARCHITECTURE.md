# Arquitectura — Previsión de demanda eléctrica en España a 24 h

Complementa al [README](README.md): describe cómo está organizado el sistema y los contratos de datos entre sus piezas.
El modelo, el borde IA, la ejecución y los tests se explican en el README.

Las piezas marcadas como **(P. III)** se añaden en el Proyecto III sin cambiar el núcleo.

---

## 1. Vista de sistema

```text
 FUENTES EXTERNAS          PRODUCTOR DE DATOS            ESTE REPOSITORIO                      CLIENTE
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 Energy-Charts ─┐
                ├──► dsia-dataset ──► energy_weather_es_hourly.csv ┄┄┄► ingest ──► BD
 Open-Meteo ────┘   (build_dataset                                       (CLI)      ▲
 (archive)           --update)                                                      │
                                                                                    │
 Open-Meteo ◄──────── previsión meteo próximas 24 h ─────────┐                      │
 (forecast)                                                  │                      │
                                                         ┌───┴──────────────────────┴───┐
 NVIDIA API   ◄─────── AIClient (mock | nvidia) ─────────│ ForecastService              │
                                                         │  ▲ CLI forecast              │
                                                         │  ▲ API (P. III)              │◄────► Dashboard
                                                         └───▲──────────────────────────┘ HTTP  (P. III)
                                                             │ carga model.joblib
                              CLI train ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
                              (lee BD → entrena → guarda modelo + métricas)

 ──── en tiempo de ejecución      ┄┄┄┄ batch / offline
```

- `dsia-dataset` es un productor externo: solo se consume su CSV, nunca se importa su código.
- El CSV solo se lee en la ingesta. Después, la **base de datos es la fuente de verdad**.
- El dashboard (P. III) solo habla HTTP con la API; nunca accede a la BD ni al CSV.
- El CLI `pipeline` encadena `ingest` → `train` → `forecast` usando los exit codes como contrato
  (`scripts/run_pipeline.sh` lo lanza desde cron).

---

## 2. Vista por capas

```text
 Dashboard (P. III)
      │ HTTP
      ▼
 ┌────────────────────┐        ┌─────────────────────────────────────┐
 │ Interfaz           │───────►│               DOMINIO               │
 │ CLIs · API (P. III)│        │                                     │
 └─────────┬──────────┘        │ Contratos: schema, modelos Pydantic,│
           ▼                   │   excepciones                       │
 ┌────────────────────┐        │ Lógica pura: validator, features,   │
 │ Servicios          │───────►│   modeling, predictor, evaluation,  │
 └─────────┬──────────┘        │   summary                           │
           ▼                   │                                     │
 ┌────────────────────┐        │ (no depende de nada)                │
 │ Infraestructura    │───────►│                                     │
 └─────────┬──────────┘        └─────────────────────────────────────┘
           ▼
   BD · CSV · Open-Meteo · NVIDIA API · fichero del modelo
```

| Capa | Contiene | Responsabilidad | No puede |
|---|---|---|---|
| **Interfaz** | CLIs `ingest`, `train`, `forecast`, `pipeline`; rutas FastAPI (P. III) | Validar argumentos/peticiones, llamar a un servicio, elegir exit code o código HTTP, escribir informes | Lógica de negocio, leer CSV, SQL |
| **Servicios** | `IngestionService`, `TrainingService`, `ForecastService`; `MonitoringService` (P. III) | Orquestar: pedir datos a la infraestructura, pasarlos al dominio y guardar resultados | Conocer HTTP o detalles de la BD (usa `Protocol`) |
| **Dominio** | Contratos y lógica pura | Transformar DataFrame → DataFrame / número / modelo | Cualquier I/O |
| **Infraestructura** | Adaptadores de I/O detrás de un `Protocol` | Leer y escribir en BD, disco y red | Decisiones de negocio |

`bootstrap.py` compone el sistema: elige las implementaciones concretas a partir de `Settings` (`.env`).

### Adaptadores de infraestructura

| `Protocol` | Implementación real | Fake para tests | Usado por |
|---|---|---|---|
| `DatasetSource` | `CsvDatasetSource` (ruta o URL en `DATASET_PATH`) | DataFrame en memoria | Ingesta |
| `ObservationRepository` | `SqlObservationRepository` | `InMemoryObservationRepository` | Ingesta, entrenamiento, previsión, monitorización |
| `PredictionRepository` | `SqlPredictionRepository` | `InMemoryPredictionRepository` | Previsión, monitorización |
| `WeatherForecastClient` | `OpenMeteoForecastClient` | `FakeWeatherForecastClient` | Previsión |
| `AIClient` | `NvidiaClient` | `MockClient` (por defecto) | Previsión (explicación) |
| `ModelStore` | `JoblibModelStore` | Modelo pequeño en fixture | Entrenamiento, previsión |

---

## 3. Casos de uso

### Ingesta

```text
CLI ingest
  └─► IngestionService.run()
        ├─► DatasetSource.read()                    lee el CSV de dsia-dataset
        ├─► validate(df)                            contrato de entrada (§4.1)
        └─► ObservationRepository.upsert(df)        escribe en observations (idempotente)
```

### Entrenamiento

```text
CLI train
  └─► TrainingService.train(tune=False)
        ├─► ObservationRepository.range()           todas las observaciones
        ├─► build_training_frame(...)               mismo código de features que la previsión
        ├─► temporal_split + fit + evaluate         MAE, MAPE, RMSE
        └─► ModelStore.save(model, name, metrics)   model.joblib + reports/metrics.json
```

### Previsión a 24 h con explicación

```text
CLI forecast  (P. III: también POST /forecast)
  └─► ForecastService.forecast(origin, explain=True)
        ├─► ModelStore.load()                       primero: sin modelo falla antes de llamar a la red
        ├─► ObservationRepository.last_hours(336)   MIN_HISTORY_HOURS
        ├─► WeatherForecastClient.next_hours(origin)
        ├─► build_inference_frame(history, weather)
        ├─► predict_forecast(model, features)       → Forecast
        ├─► PredictionRepository.save(forecast)
        ├─► summarize_forecast(...)                 → ForecastSummary
        └─► AIClient.explain(summary)               mock | nvidia; si falla, mock-fallback
```

---

## 4. Contratos de datos

### 4.1 Entrada: CSV de `dsia-dataset`

Definido en [`domain/schema.py`](energy_forecast/domain/schema.py) y comprobado por `domain/validator.py`.
Si el fichero no existe o el schema es inválido, `ingest` termina con exit code `2`.

| Columna | Tipo | Notas |
|---|---|---|
| `time_utc` | datetime UTC | Una fila por hora, sin duplicados |
| `time_local` | datetime | `Europe/Madrid` |
| `load_mw` | float | Objetivo. Rango válido `15.000–50.000`; NaN permitido (apagón) |
| `temperature_2m` | float | °C, `-20–50` |
| `relative_humidity_2m` | float | %, `0–100` |
| `wind_speed_10m`, `shortwave_radiation`, `precipitation` | float | ≥ 0 |
| `is_holiday`, `covid_flag`, `load_interpolated`, `weather_interpolated` | int (0/1) | Indicadores |

Las filas con `load_mw` NaN o fuera de rango se guardan (sirven para los lags) pero no se usan como objetivo.

### 4.2 Salida: previsión (`Forecast`)

Modelo Pydantic inmutable en [`domain/models.py`](energy_forecast/domain/models.py). Es la forma de
`reports/forecast_<origen>.json` y de la respuesta de `/forecast` (P. III).

| Campo | Tipo | Notas |
|---|---|---|
| `origin` | datetime UTC | Última hora con demanda real conocida |
| `predictions` | 24 × `{time_utc, predicted_mw}` | Horas UTC consecutivas desde `origin + 1 h`; `predicted_mw ≥ 0` |
| `model_version` | str | p. ej. `hgb-20261112T001643Z` |
| `explanation` | str \| null | `null` con `--no-explain` |
| `provider` | `mock` \| `nvidia` \| `mock-fallback` \| null | |
| `ai_model` | str \| null | `null` con el mock |

En la API (P. III): `400/422` entrada inválida, `502/503` proveedor externo caído, `500` error inesperado.

### 4.3 Borde IA: `ForecastSummary`

Lo calcula el dominio (`domain/summary.py`) y es lo único que recibe `AIClient.explain`: el LLM redacta, no calcula.

| Campo | Tipo |
|---|---|
| `period_start_local`, `period_end_local` | datetime (hora local) |
| `peak_time_local`, `peak_mw`, `valley_time_local`, `valley_mw`, `mean_mw` | datetime / float |
| `change_vs_last_week_pct` | float \| null (sin datos de la semana anterior) |
| `mean_temperature_c` | float |
| `is_holiday`, `is_weekend` | bool (alguna hora del periodo) |
| `model_mae_mw` | float \| null |

### 4.4 Resultados de entrenamiento

`TrainingResult` (`domain/models.py`) se serializa en `reports/metrics.json`: `model_version`, `best_model`,
`validation_metrics` y `metrics` (test) por modelo (`{mae, rmse, mape}`), `mae_by_hour`, periodos de
entrenamiento y test, y `features`. `IngestionReport` resume una ingesta (`rows_in`, `rows_valid`,
`rows_rejected`, `rows_written`).

### 4.5 Base de datos

SQLite (definida en `infrastructure/db.py` con SQLAlchemy). En el P. III puede pasar a Postgres sin tocar más que la
infraestructura.

| Tabla | Columnas | Escribe | Lee |
|---|---|---|---|
| `observations` | `time_utc` (PK), `load_mw` (nullable), variables meteo, indicadores | Ingesta | Entrenamiento, previsión, monitorización, `/history` |
| `predictions` | `run_id` + `target_time_utc` (PK), `created_at`, `origin`, `predicted_mw`, `model_version` | Previsión | Monitorización, `/history` |
| `pipeline_runs` | `run_id` (PK), `kind`, `status`, `started_at`, `finished_at`, `rows`, `error` | Los CLIs | `/pipeline/status` |

### 4.6 Configuración (`.env`)

`DATASET_PATH`, `DATABASE_URL`, `MODEL_PATH`, `REPORTS_DIR`, `AI_PROVIDER` (`mock` por defecto | `nvidia`),
`NVIDIA_API_KEY`, `NVIDIA_MODEL`, `LOG_LEVEL`. Plantilla en `.env.example`. Con `AI_PROVIDER=nvidia` y sin clave, la
configuración es inválida (exit code `2`).

---

## 5. Estructura de paquetes

```text
proyecto_ii/
  energy_forecast/
    domain/            # schema, models, errors, validator, features, modeling, evaluation, predictor, summary
    services/          # ingestion, training, forecast, monitoring (P. III)
    infrastructure/    # dataset_source, db, repositories, weather_client, ai_client, model_store
    cli/               # ingest, train, forecast, pipeline
    api/               # FastAPI: app, dependencies, schemas, routers (P. III)
    bootstrap.py       # composición a partir de Settings
    config.py          # lectura de .env
    logging_config.py
  dashboard/           # P. III: solo consume la API
  models/              # model.joblib (no versionado)
  reports/             # metrics.json, forecast_<origen>.json
  scripts/             # run_pipeline.sh (cron)
  tests/               # domain/ services/ infrastructure/ cli/ + fakes.py, conftest.py, test_integration.py
```
