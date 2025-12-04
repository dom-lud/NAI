# Klasyfikacja zbioru danych Ecoli — Drzewo decyzyjne i SVM

**Autorzy:**  
Dominik Ludwiński  
Bartosz Dembowski

---

## Cel zadania
Celem projektu była klasyfikacja danych z zestawu **Ecoli (UCI Machine Learning Repository)** z wykorzystaniem dwóch klasycznych algorytmów:
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)** (różne typy kernel function)

Modele uczone były na **dwóch cechach numerycznych**:
- `MCG` — measure of signal sequence recognition
- `GVH` — signal peptide recognition score

Pozwala to na wizualizację 2D granic decyzyjnych.

---

## Źródło danych
Dataset Ecoli:  
https://archive.ics.uci.edu/ml/datasets/ecoli

Plik użyty w projekcie: `ecoli.csv`

Liczba próbek: **336**  
Liczba klas: **8** (`cp`, `im`, `imL`, `imS`, `imU`, `om`, `omL`, `pp`)

---

## Wizualizacja danych wejściowych

![input](./results/Ecoli%20-%20Input%20Data.png)

Widoczna jest częściowa separacja klas — dane trudne do klasyfikacji liniowej.

---

## Decision Tree Classifier

Kod: `ecoli_decision_tree.py`

### Wyniki — zbiór treningowy
![dt-train](./results/Ecoli%20-%20Decision%20Tree(Train).png)

### Wyniki — zbiór testowy
![dt-test](./results/Ecoli%20-%20Decision%20Tree(Test).png)

Metryki testowe:

| Klasa | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:------:|
| cp | 0.76 | 0.86 | 0.81 | 36 |
| im | 0.32 | 0.32 | 0.32 | 19 |
| pp | 0.64 | 0.69 | 0.67 | 13 |
| pozostałe | 0.0 | 0.0 | 0.0 | niskie wsparcie |

**Accuracy (test): 0.62**

---

### Eksperyment: wpływ max_depth

| max_depth | accuracy |
|:---------:|:--------:|
| 3 | 0.643 |
| 5 | 0.619 |
| 7 | 0.607 |
| 9 | 0.595 |
| None | 0.560 |

Wniosek:  
Zbyt głębokie drzewo **przeucza się**, spada accuracy.

---

## Support Vector Machine (SVM)

Kod: `ecoli_svm.py` oraz `ecoli_svm_kernels.py`

### Kernel: linear

Granice decyzyjne:
| Train | Test |
|-------|------|
| ![svc-train](./results/Ecoli%20-%20SVC%20linear(Train).png) | ![svc-test](./results/Ecoli%20-%20SVC%20linear(Test).png) |

**Accuracy (test): 0.56** → klasy nie są liniowo separowalne

---

## Porównanie kernel function

Zestawienie wyników ze skryptu `ecoli_svm_kernels.py`:

| Kernel | Accuracy | Macro F1 |
|--------|:--------:|:--------:|
| linear | 0.560 | 0.214 |
| RBF, C=1 | 0.679 | 0.382 |
| **RBF, C=5** | **0.702** | **0.402** |
| poly (deg=3) | 0.667 | 0.373 |
| sigmoid | 0.429 | 0.086 |

Najlepszy model: **SVM RBF (C=5)**  
potrafi uchwycić nieliniowy charakter danych

---

## Przykład klasyfikacji
Kod wykorzystuje poglądową próbkę:
example_sample = [[0.5, 0.5]]


Wynik dla Decision Tree:
> Przewidziana klasa: `cp`

---

## Wnioski końcowe
- Dane **nie są liniowo separowalne**, co tłumaczy słabe wyniki SVM z kernelem liniowym
- Najlepiej działa **SVM z RBF** z dobranym parametrem `C`
- Małe klasy (1–2 próbki) → **trudne do poprawnej klasyfikacji** → zerowe F1-score
- Decision Tree daje intuicyjne granice, ale łatwo się przeucza na tym zbiorze

---

## ▶Jak uruchomić?
W katalogu `ecoli` uruchom:

```bash
python ecoli_decision_tree.py
python ecoli_svm.py
python ecoli_svm_kernels.py
