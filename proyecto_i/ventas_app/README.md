# VENTAS APP

Parte A del Proyecto I de DSIA.

---
## Datos

### Dataset

CSV de ventas de juguete proporcionado por el curso, copiado sin cambios del repositorio de la asignatura
([`dsia-26-27`](https://github.com/dmartincc/dsia-26-27), `1_programacion_avanzada_python/Datos/ventas.csv`).

| Métrica | Valor |
|---|---|
| Fichero | [`data/ventas.csv`](../data/ventas.csv) |
| Periodo | `2026-01-05` → `2026-04-04` |
| Filas | 150 (una por venta) |
| Columnas | 6 |
| Tamaño | 5.9 KB |
| Regiones | 4 (Norte 46, Este 39, Sur 37, Oeste 28) |
| Productos | 5 (Cable Kit, Hub IoT, Sensor A, Sensor B, Gateway) |
| Clientes | 38 |

Al ser un dataset pequeño se incluye completo en el repositorio. Contiene **10 filas inválidas** a propósito:

- `unidades` vacías (2), no numéricas (`na`, 1), cero (2) o negativas (1).
- `precio_unitario` vacío (1), cero (1) o negativo (3).

(Una fila tiene a la vez `unidades = 0` y `precio_unitario = -1`.)

### Columnas

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha` | date | Fecha de la venta (`YYYY-MM-DD`) |
| `region` | str | Región de venta (Norte, Sur, Este, Oeste) |
| `producto` | str | Nombre del producto vendido |
| `unidades` | int | Unidades vendidas (llega como texto por el valor `na`) |
| `precio_unitario` | float | Precio por unidad (€) |
| `cliente_id` | str | Identificador del cliente (`C001`, `C002`, …) |

### Fuente y licencia

Dataset de ejemplo del curso, de uso docente. El repositorio de la asignatura no indica una licencia específica.

---
## Ejecución

Desde la carpeta `proyecto_i`:

```bash
python -m ventas_app.cli --input data/ventas.csv --output data/ventas_limpias.csv
```

Escribe las ventas válidas (con la columna derivada `importe = unidades * precio_unitario`) en
[`data/ventas_limpias.csv`](../data/ventas_limpias.csv) y muestra el importe por región:

```bash
Registros válidos: 140 | inválidos: 10
Importe por región:
  Este: 33122.53
  Norte: 22914.63
  Sur: 21396.35
  Oeste: 19164.66
```

---
## Resultados testing

Tests en `tests/ventas/` (loader, validator, metrics y cli). El test de integración comprueba que `ventas.csv` tiene **140 filas válidas y 10 inválidas**.

```python
pytest -q tests/ventas
```
```bash
13 passed
```

```python
pytest -q tests/ventas -m "not integration"
```
```bash
12 passed, 1 deselected
```

---
## CI

Link a workflow en verde: https://github.com/pelayosl/dsia-26-27-sierra-pelayo/actions/runs/38171852799

Workflow: `.github/workflows/ci.yml` (único para el repositorio; cubre Parte A y Parte B).

```bash
pytest -q --cov=ventas_app --cov=energy_forecast_app --cov-report=term-missing --cov-fail-under=60
```
