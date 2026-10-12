import requests
try:
    respuesta = requests.get("https://api.open-meteo.com/v1/forecast", timeout=5)
    respuesta.raise_for_status() # Lanza una excepción si la respuesta no es 2xx
    print(f'Solicitud exitosa. Código de estado: {respuesta.status_code}')
    # Aquí podrías añadir un print(respuesta.json()) para ver el contenido de la respuesta
except requests.exceptions.Timeout:
    print('La solicitud excedió el tiempo de espera.')
except requests.exceptions.RequestException as error:
    print(f'Error al consultar la API: {error}')

import requests 
import json 
 
URL = "https://api.open-meteo.com/v1/forecast" 
PARAMETROS = { 
    "latitude": -12.07,      # Huancayo 
    "longitude": -75.21, 
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", 
    "timezone": "America/Lima", 
    "forecast_days": 7, 
} 
 
try: 
    respuesta = requests.get(URL, params=PARAMETROS, timeout=5) 
    respuesta.raise_for_status() 
    datos = respuesta.json() 
except requests.exceptions.RequestException as error: 
    raise SystemExit(f'No se pudo obtener el pronóstico: {error}') 
 
print(json.dumps(datos["daily"], indent=2, ensure_ascii=False))