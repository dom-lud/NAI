"""
FuzzyFan – Sterowanie prędkością wentylatora za pomocą logiki rozmytej

Projekt:
System wykorzystuje logikę rozmytą typu Mamdaniego do sterowania prędkością wentylatora
na podstawie trzech parametrów środowiskowych: temperatury, wilgotności i jakości powietrza.

Funkcjonalność:
- Określenie zbiorów rozmytych dla temperatury, wilgotności, jakości powietrza oraz prędkości wentylatora.
- Definicja reguł sterowania wentylatorem w oparciu o aktualne warunki środowiskowe.
- Symulacja dynamicznych zmian środowiska oraz reakcji wentylatora.
- Wizualizacja wyników symulacji na wykresie z dwiema osiami Y.

Wejścia:
- temperature (°C): temperatura w zakresie 18–40
- humidity (%): wilgotność w zakresie 40–100
- air_quality (PM2.5): jakość powietrza w zakresie 0–500

Wyjście:
- fan_speed (%): prędkość wentylatora w zakresie 0–100
Autorzy:
- Dominik Ludwiński
- Bartosz Dembowski

Wymagania środowiskowe
Instalacja Python
Python 3.10+ zalecany

Instalacja zależności:
pip install scikit-fuzzy numpy matplotlib

Uruchomienie:
python fuzzyFan.py


"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


"""
    Tworzy i konfiguruje system sterowania wentylatorem oparty na logice rozmytej.

    Zbiory rozmyte:
    - temperature: low, ideal, medium, high
    - humidity: low, ideal, medium, high
    - air_quality: good, ideal, medium, bad
    - fan_speed: very_low, low, medium, high

    Reguły sterowania wentylatorem:
    - Jeśli temperatura jest wysoka lub jakość powietrza jest zła → wentylator wysoki
    - Jeśli temperatura jest średnia lub jakość powietrza jest średnia → wentylator średni
    - Jeśli wilgotność jest wysoka → wentylator średni
    - Jeśli temperatura jest idealna i wilgotność średnia i powietrze dobre → wentylator niski
    - Jeśli wszystkie warunki idealne → wentylator bardzo niski

    Returns
    -------
    ctrl.ControlSystemSimulation
        Obiekt symulacji systemu rozmytego, gotowy do użycia w symulacji.
    """
def setup_fuzzy_system():
    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    air_quality = ctrl.Antecedent(np.arange(0, 501, 1), 'air_quality')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    # ---- Temperatura ----
    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 18])
    temperature['ideal'] = fuzz.trimf(temperature.universe, [18, 20, 22])
    temperature['medium'] = fuzz.trimf(temperature.universe, [20, 26, 30])
    temperature['high'] = fuzz.trimf(temperature.universe, [28, 40, 40])

    # ---- Wilgotność ----
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['ideal'] = fuzz.trimf(humidity.universe, [40, 50, 60])
    humidity['medium'] = fuzz.trimf(humidity.universe, [55, 70, 80])
    humidity['high'] = fuzz.trimf(humidity.universe, [75, 100, 100])

    # ---- Jakość powietrza ----
    air_quality['good'] = fuzz.trimf(air_quality.universe, [0, 0, 80])
    air_quality['ideal'] = fuzz.trimf(air_quality.universe, [30, 60, 100])
    air_quality['medium'] = fuzz.trimf(air_quality.universe, [80, 150, 250])
    air_quality['bad'] = fuzz.trimf(air_quality.universe, [200, 500, 500])

    # ---- Prędkość wentylatora ----
    fan_speed['very_low'] = fuzz.trimf(fan_speed.universe, [0, 0, 15])
    fan_speed['low']      = fuzz.trimf(fan_speed.universe, [10, 20, 30])
    fan_speed['medium']   = fuzz.trimf(fan_speed.universe, [25, 35, 50])
    fan_speed['high']     = fuzz.trimf(fan_speed.universe, [45, 75, 100])

    # ---- Reguły ----
    rules = [
        # wysokie fan_speed
        ctrl.Rule(temperature['high'] | air_quality['bad'], fan_speed['high']),
        # średni fan_speed
        ctrl.Rule(temperature['medium'] & air_quality['medium'], fan_speed['medium']),
        ctrl.Rule(humidity['high'], fan_speed['medium']),
        # niski fan_speed
        ctrl.Rule(temperature['ideal'] & air_quality['medium'], fan_speed['low']),
        ctrl.Rule(temperature['medium'] & air_quality['good'], fan_speed['low']),
        ctrl.Rule(temperature['ideal'] & humidity['medium'] & air_quality['good'], fan_speed['low']),
        # bardzo niski fan_speed
        ctrl.Rule(temperature['ideal'] & humidity['ideal'] & air_quality['ideal'], fan_speed['very_low']),
        ctrl.Rule(temperature['low'] & air_quality['good'], fan_speed['very_low']),
        ctrl.Rule(temperature['low'] & humidity['ideal'], fan_speed['very_low']),
    ]

    controller = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(controller)


"""
    Przeprowadza symulację środowiska i sterowania prędkością wentylatora.

    Symulacja uwzględnia:
    - dynamiczne zmiany temperatury, wilgotności i jakości powietrza
    - wpływ prędkości wentylatora na środowisko
    - automatyczne wyłączenie wentylatora przy idealnych warunkach


        Historia zmian parametrów w postaci słownika:
        - "temp": lista wartości temperatury
        - "hum": lista wartości wilgotności
        - "pm": lista wartości jakości powietrza (PM2.5)
        - "fan": lista wartości prędkości wentylatora (%)
    """
def run_simulation():
    sim = setup_fuzzy_system()

    T_MIN, T_MAX = 18, 22
    H_MIN, H_MAX = 40, 60
    PM_MIN, PM_MAX = 0, 100

    temp = 40.0
    hum = 65.0
    pm = 160.0

    history = {"temp": [], "hum": [], "pm": [], "fan": []}

    for t in range(300):
        sim.input['temperature'] = temp
        sim.input['humidity'] = hum
        sim.input['air_quality'] = pm
        sim.compute()

        # ✅ Twarde wyłączenie wentylatora przy idealnych warunkach
        if (T_MIN <= temp <= T_MAX) and (H_MIN <= hum <= H_MAX) and (PM_MIN <= pm <= PM_MAX):
            fan = 0
        else:
            fan = sim.output['fan_speed']

        history["temp"].append(temp)
        history["hum"].append(hum)
        history["pm"].append(pm)
        history["fan"].append(fan)

        # 🔹 Bardziej realistyczny model środowiska
        temp += (22 - temp) * 0.03 - fan * 0.01
        hum += (55 - hum) * 0.02 - fan * 0.005 + np.random.uniform(-0.2, 0.2)
        pm += (80 - pm) * 0.04 - fan * 0.05

        # Ograniczenia fizyczne
        temp = np.clip(temp, 0, 40)
        hum = np.clip(hum, 0, 100)
        pm = np.clip(pm, 0, 500)

        print(f"[{t}s] Temp={temp:.2f}°C | Hum={hum:.2f}% | PM={pm:.2f} | Fan={fan:.1f}%")

        if fan == 0:
            print("\n✅ Idealne warunki osiągnięte — wentylator wyłączony.")
            break

    return history


"""
   Tworzy wykres symulacji sterowania wentylatorem z dwiema osiami Y.

   Parameters
   ----------
   
    Historia zmian parametrów zwrócona przez funkcję run_simulation().
    owinna zawierać klucze: "temp", "hum", "pm", "fan".

   Wykres:
   - Oś X: czas [s]
   - Oś Y1: temperatura [C] i wilgotność [%]
   - Oś Y2: PM2.5 oraz prędkość wentylatora [%]
   """
def plot(history):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(history["temp"], label="Temp [°C]", color='tab:red')
    ax1.plot(history["hum"], label="Hum [%]", color='tab:blue')
    ax2.plot(history["pm"], label="PM2.5", color='tab:green')
    ax2.plot(history["fan"], label="Fan [%]", color='tab:orange', linewidth=3)

    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Temp / Hum")
    ax2.set_ylabel("PM / Fan")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Symulacja sterowania wentylacją — logika rozmyta")
    plt.grid(True)
    plt.show()


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    data = run_simulation()
    plot(data)
