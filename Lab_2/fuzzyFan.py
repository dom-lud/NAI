"""
Projekt: Fuzzy Logic – sterowanie prędkością wentylatora

Opis:
System steruje prędkością wentylatora na podstawie temperatury,
wilgotności oraz jakości powietrza (PM2.5) przy użyciu logiki rozmytej Mamdaniego.

Wejścia:
- temperature (0–40 °C)
- humidity (0–100 %)
- air_quality (0–500 PM2.5)

Wyjście:
- fan_speed (0–100 %) – sterowanie prędkością wentylatora

Autor: 
Dominik Ludwiński
Bartosz Dembowski

Wymagania środowiskowe
Instalacja Python
Python 3.10+ zalecany

Instalacja zależności:
pip install scikit-fuzzy numpy matplotlib

Uruchomienie:
python fan_fuzzy.py


"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# --- Zmienne wejściowe ---
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
air_quality = ctrl.Antecedent(np.arange(0, 501, 1), 'air_quality')


# --- Zmienna wyjściowa ---
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')


# --- Funkcje przynależności ---
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 40, 40])


humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])


air_quality['good'] = fuzz.trimf(air_quality.universe, [0, 0, 100])
air_quality['medium'] = fuzz.trimf(air_quality.universe, [50, 150, 250])
air_quality['bad'] = fuzz.trimf(air_quality.universe, [200, 500, 500])


fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 40])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [60, 100, 100])


# --- Reguły rozmyte ---
r1 = ctrl.Rule((temperature['high'] & humidity['high']) | (air_quality['bad']), fan_speed['high'])
r2 = ctrl.Rule((temperature['medium'] & humidity['medium']) | (air_quality['medium']), fan_speed['medium'])
r3 = ctrl.Rule((temperature['low'] & humidity['low'] & air_quality['good']), fan_speed['low'])
r4 = ctrl.Rule((temperature['high'] & air_quality['medium']), fan_speed['high'])
r5 = ctrl.Rule((humidity['high'] & air_quality['medium']), fan_speed['medium'])
r6 = ctrl.Rule((temperature['low'] & humidity['high']), fan_speed['medium'])


# --- System ---
fan_ctrl = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6])
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)


# --- Przykład ---
fan_sim.input['temperature'] = 28
fan_sim.input['humidity'] = 65
fan_sim.input['air_quality'] = 180
fan_sim.compute()
print(f"Prędkość wentylatora: {fan_sim.output['fan_speed']:.2f}%")


# --- Wykresy przynależności ---
temperature.view(); humidity.view(); air_quality.view(); fan_speed.view();

plt.show()
