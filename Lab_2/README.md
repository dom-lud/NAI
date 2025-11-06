# Projekt: Regulator prędkości wentylatora z użyciem logiki rozmytej

## Opis projektu

System steruje prędkością wentylatora w zależności od trzech parametrów środowiskowych:

* **Temperatura** (0–40°C)
* **Wilgotność** (0–100%)
* **Jakość powietrza PM2.5** (0–500)

Wyjściem systemu jest:

* **Prędkość wentylatora** (0–100%)

System wykorzystuje logikę rozmytą do określania optymalnej prędkości wentylatora w czasie rzeczywistym.

---

## Wymagania środowiskowe

### Instalacja Python

Python 3.10+ zalecany

### Instalacja zależności

```bash
pip install scikit-fuzzy numpy matplotlib
```

---



---

## Uruchomienie

```bash
python fan_fuzzy.py
```

---

## Przykładowy wynik

```
Prędkość wentylatora: 72.35%
```

---


## Autorzy

 Dominik Ludwiński
 Bartosz Dembowski
