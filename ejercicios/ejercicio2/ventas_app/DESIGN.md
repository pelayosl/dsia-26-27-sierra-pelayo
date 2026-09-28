- Principio aplicado: SRP
  - Fichero/clase: `ventas_app/validator.py` / `SalesValidator`
  - Por qué cumple: la validación de ventas queda separada de la carga de datos y del cálculo de métricas. `SalesValidator` solo decide qué registros son válidos y devuelve los registros válidos junto con los inválidos, sin mezclar responsabilidades ni acoplar la lógica de negocio a la entrada/salida.

- Principio aplicado: DIP
  - Fichero/clase: `ventas_app/cli.py` y `ventas_app/loader.py` / `SalesRepository` + `CsvSalesRepository`
  - Por qué cumple: `cli.py` depende de la abstracción `SalesRepository`, no de una implementación concreta del CSV. Así, la orquestación del flujo no sabe ni le importa cómo se carga el archivo; puede cambiarse a `JsonSalesRepository` sin tocar la lógica del CLI.

- Principio aplicado: OCP
  - Fichero/clase: `ventas_app/metrics.py` / `SalesMetrics`
  - Por qué cumple: las métricas están encapsuladas en su propia clase, por lo que se pueden añadir nuevas métricas (por producto, por mes, etc.) sin modificar la validación ni el repositorio. La lógica existente queda estable y se extiende con nuevas funciones en vez de reescribirse.
