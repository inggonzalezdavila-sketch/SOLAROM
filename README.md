# SOLAROM

Aplicación web académica para priorizar la inspección diaria de inversores fotovoltaicos.

SOLAROM analiza datos SCADA horarios, calcula cinco variables operativas y combina un modelo **Isolation Forest** con una regla de rendimiento relativo (PR). La salida es una orden de trabajo que indica qué inversores conviene inspeccionar primero, la energía estimada en riesgo y una pista técnica para orientar la revisión.

Incluye **SOLAROM IA**, un chat interactivo que recibe el resumen del escenario activo y explica las alertas, los puntajes, la prioridad de inspección y los límites del modelo. La integración usa Puter AI desde el navegador y no guarda una clave privada en el repositorio. La primera consulta puede solicitar al visitante autorización o inicio de sesión.

## Abrir la aplicación

La versión publicada funciona directamente en el navegador mediante GitHub Pages. También puede ejecutarse localmente abriendo `index.html`.

## Cargar datos

La interfaz acepta archivos CSV con una fila por inversor y hora, preferiblemente de 06:00 a 18:00, y estas columnas:

```text
fecha_hora,inversor,ghi_wm2,t_amb_c,p_ac_kw,t_inv_c
```

## Alcance y transparencia

- Los datos de demostración son sintéticos.
- El modelo actual fue entrenado para una planta simulada de 12 inversores de 800 kW.
- Una anomalía es una señal para inspeccionar, no un diagnóstico definitivo.
- El chat envía al servicio de IA la pregunta y un resumen calculado del escenario; no envía el archivo CSV completo. No deben introducirse credenciales ni información confidencial.
- Antes de usarlo operacionalmente debe reentrenarse y validarse con datos SCADA reales de la planta objetivo.

Proyecto académico de la Especialización en Sistemas Energéticos Sostenibles de la Universidad de La Guajira.
