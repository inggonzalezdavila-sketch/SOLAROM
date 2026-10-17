# SOLAROM

Aplicación web académica para priorizar la inspección diaria de inversores fotovoltaicos.

SOLAROM analiza datos SCADA horarios, calcula cinco variables operativas y combina un modelo **Isolation Forest** con una regla de rendimiento relativo (PR). La salida es una orden de trabajo que indica qué inversores conviene inspeccionar primero, la energía estimada en riesgo y una pista técnica para orientar la revisión.

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
- Antes de usarlo operacionalmente debe reentrenarse y validarse con datos SCADA reales de la planta objetivo.

Proyecto académico de la Especialización en Sistemas Energéticos Sostenibles de la Universidad de La Guajira.
