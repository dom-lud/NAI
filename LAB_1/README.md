# Regina Placement (easyAI + Negamax)

### Autorzy
- **Dominik Ludwiński (s26964)**  
- **Bartosz Dembowski (s29602)**  

---

## Opis gry

**Regina Placement** to turowa gra logiczna dla dwóch graczy (człowiek vs AI lub AI vs AI), oparta o algorytm **adversarial search – Negamax** z biblioteki `easyAI`.

Gracze na przemian stawiają swoje **hetmany (A i B)** na planszy 5x5.  
Celem gry jest **strategiczne rozmieszczanie figur** tak, aby zablokować przeciwnika i doprowadzić do sytuacji, w której **nie może on wykonać ruchu**.

---

## Zasady gry

1. Plansza ma rozmiar **5x5**.  
2. Gracze (A i B) na zmianę stawiają swoje hetmany.  
3. **Hetman atakuje po wierszu, kolumnie i przekątnej.**  
4. Własne hetmany mogą się „widzieć” — nie blokują siebie nawzajem.  
5. Nie można postawić hetmana:
   - na zajętym polu,  
   - na polu atakowanym przez przeciwnika.  
6. Przegrywa ten gracz, który **nie ma już żadnego legalnego ruchu**.  
7. Zwycięża ten, kto wykona **ostatni legalny ruch**.

---

## Sztuczna inteligencja

AI została zaimplementowana przy pomocy algorytmu **Negamax** (wariant Minimaxu) z ustaloną głębokością przeszukiwania (`DEPTH = 6`).  
Wartość heurystyki:
- `-100` – stan przegrany (brak ruchów),
- `0` – w pozostałych przypadkach.

---

## Przygotowanie środowiska

Wymagana jest tylko biblioteka **easyAI**:

```bash
pip install easyAI
