# VENTAS APP

Parte A del Proyecto I de DSIA.

---
## Resultados testing
```python
pytest -q
```
```bash
........                                                                                                               [100%]
8 passed in 0.70s
```

```python
pytest -q -m "not integration"
```
```bash
.......                                                                                                                [100%]
7 passed, 1 deselected in 0.73s
```

---
## CI

Link a workflow en verde: https://github.com/pelayosl/dsia-26-27-sierra-pelayo/actions/runs/38171852799

Workflow: `.github/workflows/ci.yml`

```bash
pytest -q --cov=ventas_app --cov-report=term-missing --cov-fail-under=60
```
