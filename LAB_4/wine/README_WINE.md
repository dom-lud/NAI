# Analiza jakości czerwonego wina – klasyfikacja

**Autorzy:**  
Dominik Ludwiński  
Bartosz Dembowski

---

## Cel projektu

Porównanie jakości klasyfikacji:

* **Decision Tree Classifier**
* **Support Vector Machine (SVM)**

na danych **Wine Quality** (UCI Machine Learning Repository).

Zadanie: klasyfikacja jakości wina do jednej z dwóch klas:

* **0 → złe wino** (jakość ≤ 5)
* **1 → dobre wino** (jakość ≥ 6)

---

## Dane użyte w analizie

Plik:

```text
winequality-red.csv
```

Liczba próbek: **4898**
Liczba klas: **2**

Użyte cechy do wizualizacji granic:

* `alcohol`
* `volatile acidity`

---

## Jak uruchomić?

```bash
python wine_decision_tree.py
python wine_svm.py
python wine_svm_kernels.py
```

Wygenerowane wyniki zapisują się w katalogu:

```text
LAB_4/wine/results/
```

---

## Wizualizacja danych wejściowych

![Wine Input](./results/Wine%20-%20Input%20Data.png)

---

## Decision Tree
Granice decyzyjne – zbiór **treningowy**:

![Wine DT Train](./results/Wine%20-%20Decision%20Tree(Train).png)

Granice decyzyjne – zbiór **testowy**:

![Wine DT Test](./results/Wine%20-%20Decision%20Tree(Test).png)

### Metryki – zbiór testowy

| Klasa | Opis         | Precision | Recall | F1-score |  Support |
| ----: | ------------ | :-------: | :----: | :------: | :------: |
|     0 | bad wine     |    0.63   |  0.55  |   0.59   |    410   |
|     1 | good wine    |    0.79   |  0.84  |   0.81   |    815   |
|       | **Accuracy** |  **0.74** |        |          | **1225** |

Model lepiej rozpoznaje wina dobre (klasa 1), natomiast część słabszych win jest klasyfikowana jako dobre.

---

### Eksperyment: wpływ `max_depth`

Wyniki z konsoli:

Wizualizacja:

![Wine Effect Max Depth](./results/Wine%20-%20EffectOfMax_depth.png)

| max_depth |  accuracy |
| :-------: | :-------: |
|     2     |   0.731   |
|   **4**   | **0.743** |
|     6     |   0.736   |
|     8     |   0.734   |
|    None   |   0.713   |

**Wniosek:** zbyt głębokie drzewo zaczyna się przeuczać – dokładność na zbiorze testowym delikatnie spada.

---

## SVM – kernel liniowy

Model:

```python
SVC(kernel="linear", C=1.0, decision_function_shape="ovr")
```

Granice decyzyjne – zbiór **treningowy**:

![Wine SVC lin Train](./results/Wine%20-%20SVC%20linear(Train).png)

Granice decyzyjne – zbiór **testowy**:

![Wine SVC lin Test](./results/Wine%20-%20SVC%20linear(Test).png)

### Metryki – SVC (linear)

| Zbiór | Accuracy |
| ----- | :------: |
| Train |   0.74   |
| Test  | **0.75** |

SVM z kernelem liniowym daje bardzo stabilne wyniki – dokładności na zbiorze treningowym i testowym są zbliżone, co sugeruje brak silnego przeuczenia.

---

## SVM – porównanie różnych kernel function

Dodatkowy skrypt `wine_svm_kernels.py` porównuje kilka konfiguracji SVM na tych samych cechach (`alcohol`, `volatile acidity`).

Podsumowanie z konsoli (zbiór testowy):

```text
Konfiguracja: linear, C=1.0
  accuracy (test):   0.749
  macro F1 (test):   0.684
--------------------------------------------------
Konfiguracja: RBF, C=1.0, gamma='scale'
  accuracy (test):   0.715
  macro F1 (test):   0.629
--------------------------------------------------
Konfiguracja: RBF, C=5.0, gamma='scale'
  accuracy (test):   0.725
  macro F1 (test):   0.654
--------------------------------------------------
Konfiguracja: RBF, C=1.0, gamma=0.5
  accuracy (test):   0.743
  macro F1 (test):   0.670
--------------------------------------------------
Konfiguracja: poly, degree=3, C=1.0
  accuracy (test):   0.750
  macro F1 (test):   0.670
--------------------------------------------------
Konfiguracja: poly, degree=4, C=1.0
  accuracy (test):   0.750
  macro F1 (test):   0.685
--------------------------------------------------
Konfiguracja: sigmoid, C=1.0, gamma='scale'
  accuracy (test):   0.705
  macro F1 (test):   0.665
--------------------------------------------------
```

Zbiorcza tabela:

| Kernel / konfiguracja         |  Accuracy |  F1_macro |
| ----------------------------- | :-------: | :-------: |
| linear, C=1.0                 |   0.749   |   0.684   |
| RBF, C=1.0, gamma='scale'     |   0.715   |   0.629   |
| RBF, C=5.0, gamma='scale'     |   0.725   |   0.654   |
| RBF, C=1.0, gamma=0.5         |   0.743   |   0.670   |
| poly, degree=3, C=1.0         |   0.750   |   0.670   |
| **poly, degree=4, C=1.0**     | **0.750** | **0.685** |
| sigmoid, C=1.0, gamma='scale' |   0.705   |   0.665   |

**Wnioski:**

* dane są częściowo separowalne liniowo – kernel `linear` już daje dobry wynik,
* lekkie nieliniowości poprawia kernel wielomianowy (`poly`), szczególnie stopnia 4,
* kernel RBF działa poprawnie, ale w tej konfiguracji jest minimalnie gorszy,
* kernel `sigmoid` daje najsłabszy wynik spośród testowanych.

---

## Podsumowanie dla zbioru Wine

* Oba modele (Decision Tree i SVM) osiągają **zbliżone accuracy ~0.74–0.75**.
* Drzewo decyzyjne jest łatwiejsze do interpretacji – widać proste reguły oparte na `alcohol` i `volatile acidity`.
* SVM (szczególnie z kernelem `poly`) lepiej wykorzystuje subtelne zależności i daje minimalnie wyższe metryki.
* Wina o wyższej zawartości alkoholu i niższej kwasowości lotnej są zazwyczaj klasyfikowane jako **dobre (klasa 1)**.

Wszystkie wymagane elementy zadania (wizualizacje, metryki, porównanie kernel function, przykładowe scenariusze) zostały zrealizowane na zbiorze **Wine Quality**.
