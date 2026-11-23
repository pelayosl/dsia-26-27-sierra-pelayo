# ENERGY FORECAST — Previsión de demanda eléctrica a 24 h

Proyecto II de DSIA (base del Proyecto III). Arquitectura: [`ARCHITECTURE.md`](ARCHITECTURE.md).

Recorrido del proyecto:

```text
dsia-dataset (CSV) ──► ingest ──► SQLite ──► train (sklearn) ──► models/model.joblib + reports/metrics.json
                                    │                                      │
                                    └──────────► forecast ◄────────────────┘
                                                  │  + previsión meteo (Open-Meteo)
                                                  │  + explicación (mock | NVIDIA API Catalog)
                                                  ▼
                                    reports/forecast_<fecha>.json
```

---
## Dominio y motivación

El sistema predice la **demanda eléctrica horaria de España (MW) para las próximas 24 horas**.

- **Operación de la red.** El operador del sistema tiene que casar generación y demanda en cada instante. Saber con un
  día de antelación cuánta demanda habrá en cada hora permite programar qué centrales arrancan, cuánta reserva se
  necesita y cuándo llegan las rampas de la mañana y de la tarde.
- **Mercado diario.** Comercializadoras y generadores pujan el día anterior, hora a hora, en el mercado diario de
  electricidad. Un error en la previsión de demanda se paga después en los mercados de ajuste, más caros.
- **Por qué es un buen problema de ML.** La demanda tiene patrones claros (hora del día, día de la semana, festivos,
  temperatura) pero también ruido y cambios de tendencia (autoconsumo solar, COVID, el apagón de abril de 2025), con
  datos abiertos de varios años.

Un horizonte de 24 h es el del mercado diario y permite una estrategia directa con un único modelo (ver "Modelo").
Este proyecto es la base del Trabajo Final (Proyecto III), que añadirá la API, el dashboard y el despliegue.

## Datos

El CSV de entrada es `energy_weather_es_hourly.csv`, generado por [`dsia-dataset`](https://github.com/pelayosl/dsia-dataset):
la demanda eléctrica horaria de España (Energy-Charts) combinada con meteorología (Open-Meteo, media ponderada de 5 ciudades).

- **Periodo:** desde el 1 de enero de 2019 (UTC) hasta la última hora con demanda disponible al generarlo.
- **Filas:** una por hora, ordenadas por `time_utc` y sin huecos (unas 68.000 filas hasta octubre de 2026).
- **Columnas:** `time_utc`, `time_local`, `load_mw` (objetivo), 5 variables meteorológicas e indicadores
  (`is_holiday`, `covid_flag`, `load_interpolated`, `weather_interpolated`). El contrato está en
  [`energy_forecast/domain/schema.py`](energy_forecast/domain/schema.py) y la descripción completa en el README de `dsia-dataset`.

### Cómo obtener el CSV

El CSV **no está en este repositorio** (pesa unos 7 MB y se regenera). Se genera con `dsia-dataset`:

```bash
git clone https://github.com/pelayosl/dsia-dataset.git
cd dsia-dataset
uv sync
uv run python -m src.build_dataset --start 2019-01-01   # generación completa (20-40 min la primera vez)
uv run python -m src.build_dataset --update             # después, solo las horas nuevas
```

El resultado queda en `dsia-dataset/data/output/energy_weather_es_hourly.csv`. Para usarlo, apunta `DATASET_PATH` del `.env`
a ese fichero. Las rutas relativas se resuelven desde `proyecto_ii/`. Con los dos repositorios clonados en la misma carpeta:

```dotenv
DATASET_PATH=../../dsia-dataset/data/output/energy_weather_es_hourly.csv
```

Así, cada `--update` de `dsia-dataset` se refleja en la siguiente ejecución de `ingest`, sin copiar ficheros.
Alternativas: copiar el CSV a `proyecto_ii/data/` (la ruta por defecto, ignorada por git) o poner en `DATASET_PATH`
una URL `http(s)://` al CSV.

Hace falta el dataset **completo**: la partición temporal entrena con 2019-2024, valida con 2025 y evalúa con 2026.
La muestra de un año del Proyecto I no cubre esos periodos.

Los tests **no necesitan el CSV completo**: generan datos sintéticos en memoria (fixtures de `tests/conftest.py`) y
los escriben en carpetas temporales. Solo los tests de integración del validador y del pipeline completo
(`tests/test_integration.py`) leen la muestra real de un año del Proyecto I
(`proyecto_i/data/energy_weather_es_hourly_sample.csv`, abril de 2025 a marzo de 2026). La muestra se reutiliza
desde `proyecto_i/` en lugar de copiarla a `proyecto_ii/data/`: así hay un único fichero de muestra en el repositorio
y `data/` queda solo para el dataset completo y la base de datos locales, que no se versionan.

Licencias:

- Energy-Charts (Fraunhofer ISE), CC BY 4.0, con datos de ENTSO-E sujetos a sus condiciones de uso.
- Open-Meteo, CC BY 4.0; la API gratuita es solo para uso no comercial.

## Modelo

Regresión supervisada sobre serie temporal con scikit-learn.

### Estrategia: directa con lags ≥ 24 h

A partir de un origen T (última hora con demanda real conocida) se predicen las 24 horas siguientes. Todas las
features usan lags de al menos 24 h, así que son conocidas en T: **un único modelo** predice las 24 filas de una vez,
sin realimentar sus propias predicciones (sin estrategia recursiva ni 24 modelos).

### Features

| Grupo | Features |
|---|---|
| Lags | `lag_24`, `lag_48`, `lag_168` |
| Medias móviles | Media de las 24 h y de las 168 h anteriores, con la ventana terminando como muy tarde en T |
| Calendario (hora local) | `hour`, `weekday`, `month` (categóricas), `is_holiday` (festivos nacionales), `is_weekend` |
| Meteorología de la hora objetivo | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `shortwave_radiation`, `precipitation` |
| Otros | `covid_flag` |

El entrenamiento y la previsión usan **el mismo código** (`domain/features.py`) para construir las features, así no
hay diferencias entre lo que ve el modelo al entrenar y al predecir. Se entrena con la meteorología **observada** y se
predice con la **previsión** de Open-Meteo: es una fuente de error adicional en producción (ver "Limitaciones conocidas").

Las filas con `load_mw` a NaN (apagón del 28-29 de abril de 2025) o fuera de rango no se usan como objetivo, pero se
conservan para calcular los lags. El periodo COVID se mantiene, con `covid_flag` como feature.

### Modelos comparados y partición temporal

| Modelo | Clase | Papel |
|---|---|---|
| `naive` | Naive estacional: predice `lag_168` (la misma hora de la semana anterior) | Suelo mínimo |
| `ridge` | `Ridge` con one-hot del calendario y *splines* sobre la temperatura | Referencia interpretable |
| `hgb` | `HistGradientBoostingRegressor` con el calendario como categóricas | Candidato principal |

Partición **temporal, nunca aleatoria**: train 2019-2024, validación 2025, test 2026. Cada candidato se entrena dos
veces:

1. con train, midiendo en validación: **el modelo se elige por el MAE de validación**;
2. con train + validación, midiendo en test: son las métricas que se reportan. El test nunca interviene en la elección.

El modelo guardado en `models/model.joblib` es el elegido, entrenado con train + validación (2019-2025), y es
exactamente el que describe `reports/metrics.json`. `train --tune` ajusta antes los hiperparámetros de Ridge y HGB con
`GridSearchCV` + `TimeSeriesSplit(3)` sobre train + validación.

### Métricas

Resultados de [`reports/metrics.json`](reports/metrics.json) (modelo `hgb-20261112T001643Z`, dataset completo hasta el
11 de octubre de 2026). Test: del 1 de enero al 11 de octubre de 2026.

| Modelo | MAE validación (MW) | MAE test (MW) | RMSE test (MW) | MAPE test |
|---|---|---|---|---|
| `naive` | 1.412 | 1.424 | 2.121 | 5,09 % |
| `ridge` | 927 | 962 | 1.323 | 3,51 % |
| **`hgb`** (guardado) | **611** | **640** | **937** | **2,28 %** |

El HGB reduce el MAE del baseline un 55 % y el de Ridge un 33 %. El error de validación (2025) y el de test (2026)
son parecidos, así que no hay señales de sobreajuste a la validación. El MAE por hora del día del modelo final
(`mae_by_hour`) va de 360 MW a las 4:00 a 929 MW a las 13:00; la causa se analiza en "Limitaciones conocidas".

El modelo `.joblib` no se versiona (`*.joblib` está en `.gitignore`): se regenera con `train` a partir del dataset.
`reports/metrics.json` sí se versiona como evidencia de las métricas anteriores.

## Borde IA (NVIDIA API Catalog)

Después de la predicción sklearn, un LLM redacta una **explicación breve en lenguaje natural** de las 24 predicciones
para una persona. El LLM **no sustituye** al modelo sklearn ni calcula cifras:

1. El dominio calcula un `ForecastSummary` (`domain/summary.py`): periodo en hora local, pico y valle, demanda media,
   variación frente a las mismas horas de la semana anterior, temperatura media prevista, festivo o fin de semana y
   MAE del modelo en test, todo ya redondeado.
2. `AIClient.explain(summary)` solo redacta ese resumen (prompt de sistema: *"Use only the data provided and do not
   invent figures"*).

| Campo | Valor |
|---|---|
| Proveedor | **NVIDIA API Catalog** (`https://integrate.api.nvidia.com/v1`, compatible con OpenAI) |
| Cliente | `huggingface_hub.InferenceClient(base_url=..., api_key=NVIDIA_API_KEY)`, el mismo del curso |
| Tarea | **`chat_completion`** |
| Modelo | **`nvidia/nemotron-3.5-super-vl-120b-a12b`** (configurable con `NVIDIA_MODEL`) |
| Parámetros | `max_tokens=200`, `temperature=0.3`, razonamiento desactivado (`enable_thinking=False`), *timeout* de 20 s |
| Por defecto, tests y CI | **`mock`** (`AI_PROVIDER=mock`): plantilla determinista, sin red ni claves |

**Por qué NVIDIA y no Hugging Face.** El enunciado admite "otro proveedor API documentado". En noviembre de 2026 el
modelo del curso (`mistralai/Mistral-7B-Instruct-v0.3`) no tenía ningún proveedor activo en Hugging Face y las cuentas
gratuitas de HF ya no incluyen créditos para Inference Providers. El catálogo de NVIDIA es gratuito para prototipos y se
usa con el mismo `InferenceClient` cambiando solo `base_url`.
Condiciones: solo para prototipos, investigación y pruebas, con límite de peticiones por minuto.

**Activar el proveedor real** en el `.env`:

```dotenv
AI_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...    # de https://build.nvidia.com, nunca en el repositorio
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
```

Con `AI_PROVIDER=nvidia` y sin clave, la configuración es inválida y los CLIs terminan con exit code `2`.

**El LLM no está en el camino crítico.** Si NVIDIA falla (red, *timeout*, 401, 429, 5xx, respuesta vacía o cortada),
la previsión no falla: se registra el error en el log y se usa la explicación del mock con `provider: "mock-fallback"`.
Con `forecast --no-explain` no se llama al proveedor.

## Instalación

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env   # y ajusta DATASET_PATH (ver "Cómo obtener el CSV")
```

## Cómo ejecutar

Desde `proyecto_ii/`, en este orden (no hace falta ninguna clave: por defecto `AI_PROVIDER=mock`):

```bash
python -m energy_forecast.cli.ingest     # CSV → validación → tabla observations (idempotente)
python -m energy_forecast.cli.train      # BD → 3 modelos → models/model.joblib + reports/metrics.json
python -m energy_forecast.cli.forecast   # últimas 336 h + meteo prevista → 24 predicciones + explicación
```

| CLI | Opciones | Exit codes |
|---|---|---|
| `ingest` | `--input RUTA_O_URL` (sustituye `DATASET_PATH`) | `0` ok · `2` fichero inexistente o schema inválido · `1` error inesperado |
| `train` | `--tune` (búsqueda de hiperparámetros, más lenta) | `0` ok · `2` sin datos suficientes para train/validación/test · `1` error inesperado |
| `forecast` | `--origin 2026-10-11T19:00:00+00:00` (por defecto, la última hora con demanda), `--no-explain` | `0` ok · `2` sin modelo, sin historia suficiente u origen inválido · `1` Open-Meteo caído o error inesperado |
| `pipeline` | `--train auto\|always\|never`, `--no-explain` (ver "Automatización") | `0` ok · el código del primer paso que falle |

Todos aceptan `--log-file RUTA` (copia del log en un fichero) y `-v` (nivel `DEBUG`). Cada ejecución queda registrada
en la tabla `pipeline_runs` (tipo, estado, filas y error).

`forecast` necesita que el origen esté dentro del rango que sirve Open-Meteo (unos 3 meses hacia atrás y 15 días hacia
delante). Para predecir "mañana" basta con actualizar los datos (`build_dataset --update` en `dsia-dataset` +
`ingest`); no hace falta reentrenar.

### Automatización

**Orquestador.** `python -m energy_forecast.cli.pipeline` ejecuta los pasos en orden y usa sus exit codes como
contrato entre ellos: un paso solo corre si el anterior devolvió `0`, y el pipeline devuelve el código del paso que
falló. Así nunca se publica una previsión hecha con una ingesta fallida.

```text
ingest ──0──► train (si hace falta) ──0──► forecast ──0──► exit 0
   └─≠0─ STOP         └─≠0─ STOP              └─≠0─ STOP (exit = código del paso)
```

- `--train auto` (por defecto): entrena solo si no hay modelo guardado. Entrenar es el paso lento y el modelo no
  necesita cambiar cada día.
- `--train always`: reentrena en cada ejecución. `--train never`: nunca entrena.
- Cada paso sigue registrándose en `pipeline_runs`; con `--log-file` todos escriben en el mismo log.

**Ejecución desatendida.** [`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) envuelve el orquestador para cron
(`set -euo pipefail`, log diario en `logs/pipeline_<fecha>.log`, que no se versiona). Si se define
`DSIA_DATASET_DIR`, antes actualiza el dataset con `build_dataset --update`; si eso falla, no ejecuta el pipeline.

```bash
PYTHON=.venv/bin/python DSIA_DATASET_DIR=../../dsia-dataset bash scripts/run_pipeline.sh
```

Línea de crontab para ejecutarlo cada día a las 06:15 (rutas absolutas, porque cron no carga tu entorno):

```text
15 6 * * * PYTHON=/ruta/proyecto_ii/.venv/bin/python DSIA_DATASET_DIR=/ruta/dsia-dataset bash /ruta/proyecto_ii/scripts/run_pipeline.sh
```

En Windows, el equivalente es una tarea del Programador de tareas que ejecute
`ruta\.venv\Scripts\python.exe -m energy_forecast.cli.pipeline --log-file logs\pipeline.log` con `proyecto_ii` como
carpeta de inicio (o el script anterior con el `bash.exe` de Git for Windows).

El orquestador es deliberadamente simple: un script y una línea de cron, sin Airflow ni similares. CI y cron
tienen papeles distintos: el CI comprueba el código en cada push; cron ejecuta la corrida de datos.

### Ejemplo de salida

`reports/forecast_<origen en UTC>.json` sigue el contrato de salida de [`ARCHITECTURE.md`](ARCHITECTURE.md) (§4.2), con la misma
forma que tendrá la respuesta de `/forecast` en el Proyecto III. Ejemplo real con el proveedor NVIDIA (recortado a 3 de
las 24 predicciones):

```json
{
  "origin": "2026-10-11T19:00:00Z",
  "predictions": [
    {"time_utc": "2026-10-11T20:00:00Z", "predicted_mw": 25237.934343697445},
    {"time_utc": "2026-10-11T21:00:00Z", "predicted_mw": 23313.47730429577},
    {"time_utc": "2026-10-11T22:00:00Z", "predicted_mw": 21130.50553684999}
  ],
  "model_version": "hgb-20261112T001643Z",
  "explanation": "The demand forecast covers the period from October 11, 2026, at 22:00 to October 12, 2026, at 21:00 local time. Peak demand is expected to reach 26,284 MW at 21:00 on October 12, while the valley is projected at 18,783 MW at 04:00 on the same day. The mean demand is forecast at 21,874 MW, representing a 19.2% decrease compared to the previous week. This forecast occurs during a holiday and weekend with a mean temperature of 20.2°C and a model mean absolute error of 640 MW.",
  "provider": "nvidia",
  "ai_model": "nvidia/nemotron-3.5-super-vl-120b-a12b"
}
```

Con el mock, `provider` es `"mock"`, `ai_model` es `null` y la explicación es una plantilla en inglés con las mismas
cifras. Con `--no-explain`, los tres campos son `null`. El CLI imprime además las 24 predicciones en hora local.

Los informes `forecast_*.json` no se versionan: se regeneran con cada ejecución.

## Tests

```bash
pytest -q                                   # todos, incluidos los de integración
pytest -q -m "not integration"              # lo que corre en CI
pytest -q -m "not integration" --cov=energy_forecast --cov-report=term-missing --cov-fail-under=60
```

| Tipo | Qué cubre |
|---|---|
| Dominio | `validator`, features (sin fugas del futuro), métricas, modelos candidatos y partición, inferencia de 24 h, resumen |
| Modelo sklearn | Carga e inferencia de un modelo pequeño entrenado en un fixture (`small_trained_model` + `JoblibModelStore`) |
| Servicios | Ingesta, entrenamiento y previsión de extremo a extremo con *fakes* (`tests/fakes.py`) de todos los `Protocol` |
| Infraestructura | Repositorios SQL sobre SQLite en `tmp_path` (y los mismos tests de contrato contra los *fakes*), CSV, `JoblibModelStore`, Open-Meteo y NVIDIA con clientes HTTP falsos |
| CLIs | Exit codes y salidas de `ingest`, `train` y `forecast` con la configuración apuntando a `tmp_path`; orquestador `pipeline` (orden de los pasos, parada en el primer fallo, política de entrenamiento y una ejecución completa con los pasos reales) |
| **Integración** (`-m integration`) | Pipeline completo `ingest → train → forecast` con los CLIs reales sobre la muestra real (`tests/test_integration.py`); validación de la muestra; llamadas reales a Open-Meteo y NVIDIA |

Los tests de integración de Open-Meteo y NVIDIA necesitan red (y `NVIDIA_API_KEY` el segundo, que se salta si no
está). El del pipeline completo no usa red: sustituye Open-Meteo por un cliente falso y usa el mock de IA, y adapta la
partición temporal a la muestra (train abril-octubre de 2025, validación noviembre-diciembre, test 2026). Con solo siete
meses de entrenamiento no comprueba qué modelo gana, sino que el pipeline funciona y produce 24 predicciones plausibles.

**CI** (`.github/workflows/ci.yml`, job `test-proyecto-ii`): Python 3.12, `AI_PROVIDER=mock`, sin claves, ejecuta
`pytest -m "not integration"` con un umbral de cobertura del 60 % (`--cov-fail-under=60`). La cobertura de
`energy_forecast` en el Proyecto II es del 99 % (346 tests).

## Limitaciones conocidas

### Mayor error a mediodía (autoconsumo solar y radiación agregada)

El MAE por hora del modelo final (`mae_by_hour` en `reports/metrics.json`) no es uniforme: es mínimo de madrugada
y máximo a mediodía. Valores en test (2026) con el HGB entrenado en 2019-2025:

| Hora local | MAE (MW) | Demanda media (MW) | MAPE | Sesgo medio (MW) |
|---|---|---|---|---|
| 4:00 | 360 | 22.713 | 1,60 % | −20 |
| 13:00 | 929 | 29.443 | 3,24 % | +112 |
| 21:00 | 670 | 32.588 | 2,04 % | −236 |

No se debe a la escala: a las 21:00 la demanda es mayor y el error menor, y el MAPE también es máximo a mediodía.
La causa más probable es el **autoconsumo solar**. La demanda de red no incluye la energía que producen y consumen
las placas de los tejados, así que con sol la demanda de mediodía baja. Las grandes plantas fotovoltaicas no
influyen en la demanda, porque vierten a la red. Hay dos efectos:

- **Tendencia (sesgo).** Cada año hay más autoconsumo y el pico de mediodía se aplana. La relación entre la demanda
  media de las 13:00 y la de las 4:00 baja de 1,42 en 2019 a 1,30 en 2026. El modelo aprende de años con menos
  autoconsumo y en 2026 sobreestima el mediodía (+112 MW a las 13:00). Es un efecto pequeño frente al MAE.
- **Nubosidad (varianza), lo que más pesa.** En las horas de 11:00 a 15:00 el MAE es de 1.047 MW con la radiación
  más baja (tercil inferior) y de 709 MW con la más alta. La meteorología de `dsia-dataset` es la **media
  ponderada por población de 5 ciudades** (Madrid 0,30, Barcelona 0,25, Valencia, Sevilla y Bilbao 0,15).
  Esta ponderación es adecuada para la temperatura (calefacción y aire acondicionado siguen a la población),
  pero no para la radiación:
  - los pesos no reflejan dónde está instalado el autoconsumo: el sur y el este pesan poco;
  - una media no distingue entre nubes en el sur y nubes en el norte;
  - zonas como Murcia, Aragón, las dos Castillas o Extremadura no están representadas.

Además, el modelo se **evalúa con la radiación observada**, pero en producción **predice con la radiación
prevista** por Open-Meteo, y la nubosidad es de lo más difícil de prever. Por eso el error real de mediodía
será mayor que el de la tabla.

También hay sesgo negativo en las rampas de la mañana (8:00) y de la tarde (19:00-21:00): el modelo se queda
corto cuando la demanda sube deprisa.

## Qué se ampliará en el Proyecto III

API FastAPI (`uvicorn energy_forecast.api.app:app`), dashboard, monitorización y despliegue.

Posibles mejoras del modelo, a partir de las limitaciones anteriores:

- **Radiación por ciudad como features separadas** (`shortwave_radiation_sevilla`, `_madrid`…) en lugar de la media
  ponderada, para que el modelo aprenda qué zonas importan. Requiere ampliar el contrato del CSV de `dsia-dataset`
  y que el cliente de previsión meteorológica pida cada ciudad.
- **Pesos de la radiación según la potencia de autoconsumo instalada** por comunidad en lugar de la población.
  Es más sencillo que la opción anterior, pero también cambia `dsia-dataset`.
- **Más puntos meteorológicos** en zonas solares sin cubrir (Murcia, Zaragoza, Badajoz, Toledo…).
- **Una feature de tendencia** (año o días desde el inicio), o **más peso a los años recientes** al entrenar
  (`sample_weight`), para recoger el aumento progresivo del autoconsumo.

Antes de cambiar el contrato del dataset, conviene medir cada mejora en un experimento aparte, comparando el MAE de
mediodía (`mae_by_hour`) con el modelo actual.
