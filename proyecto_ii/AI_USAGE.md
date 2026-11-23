# Uso de IA

En este documento se recogen las herramientas de inteligencia artificial generativa empleadas en este sistema.

## IA dentro del sistema

| Campo | Valor |
|---|---|
| Proveedor | **NVIDIA API Catalog** (`https://integrate.api.nvidia.com/v1`), llamado con `huggingface_hub.InferenceClient(base_url=...)` |
| Tarea | `chat_completion` |
| Modelo | `nvidia/nemotron-3.5-super-vl-120b-a12b` (configurable con `NVIDIA_MODEL`) |
| Para qué | Explicar en lenguaje natural la previsión de 24 h a partir de un resumen numérico (`ForecastSummary`) calculado por el dominio. El LLM solo redacta, no calcula cifras ni sustituye al modelo sklearn |
| Modo en tests y CI | `mock` (`AI_PROVIDER=mock`, sin `NVIDIA_API_KEY`): plantilla determinista, sin red |
| Si el proveedor falla | La previsión no falla, se usa la explicación del mock |

**Por qué no un modelo del Hub de Hugging Face.** El modelo del curso (`mistralai/Mistral-7B-Instruct-v0.3`) no tenía
ningún proveedor de inferencia activo en Hugging Face (noviembre de 2026) y las cuentas gratuitas ya no incluyen
créditos para Inference Providers. NVIDIA API Catalog es gratuito para prototipos, es compatible con la API de OpenAI y permite usar el mismo `InferenceClient` del curso.

**Qué se ajustó tras probarlo con llamadas reales:**

- Nemotron es un modelo con razonamiento. Con el razonamiento activo gastó 196 de los 200 tokens razonando y la
  respuesta salió cortada (`finish_reason=length`), así que se desactiva (`enable_thinking=False`). Sin él responde
  en 1-1,5 s.
- Una respuesta vacía o cortada se trata como fallo del proveedor y activa el *fallback* al mock.
- Se eliminan los posibles bloques `<think>...</think>` de la respuesta, por si el razonamiento se reactiva.

Condiciones de uso: gratuito con el NVIDIA Developer Program solo para prototipos, investigación y pruebas, con límite
de peticiones por minuto. La clave (`NVIDIA_API_KEY`) solo vive en el `.env`, que está en `.gitignore`.

## IA como asistente de desarrollo

Se han empleado asistentes de código para el desarrollo del proyecto ii, aunque en absoluto con la profundidad y
exhaustividad que se utilizarán en el proyecto iii. Para este proyecto, se ha utilizado con el fin de optimizar algunos
procesos, ayuda con librerías y funciones que desconocía previamente, para supervisar el cumplimiento de la arquitectura
en capas, ayuda en la implementación de tests, y para documentación del código y la elaboración del README.

Criterio general: ninguna salida de la IA se aceptó sin revisarla y ejecutar los tests.
