#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regina Placement – prosta wersja (easyAI + Negamax)
===================================================
• Plansza ma stały rozmiar 5x5 (bez zmiany z linii poleceń).
• Domyślnie "moi nie biją moich" (friendly-fire ON).
• Flaga --aivai pozwala uruchomić tryb AI kontra AI (pokaz).
• Bez flagi: gra człowieka przeciwko AI.

Jak uruchomić:
    python regina_placement_simple.py          # Człowiek vs AI
    python regina_placement_simple.py --aivai  # AI vs AI (pokaz)

Jak grać:
    - Gracze (A i B) na przemian stawiają swoje hetmany (A lub B) na planszy 5x5.
    - Ruch wykonuje się, wpisując współrzędne pola w formacie: rząd,kolumna
      np. 1,3 oznacza: wiersz 1, kolumna 3.
    - Własne hetmany (A–A, B–B) mogą się "widzieć" (nie przeszkadzają sobie).
      Zakazane są tylko konflikty między kolorami (A–B).
    - Nie można postawić hetmana na polu, które jest już zajęte lub atakowane
      przez hetmana przeciwnika.
    - Przegrywa ten gracz, który nie ma już żadnego legalnego ruchu.

Cel gry:
    - Strategicznie rozmieszczaj swoje hetmany, aby ograniczyć ruchy przeciwnika
      i zmusić go do braku możliwości wykonania ruchu.
    - Zwycięża gracz, który wykona ostatni legalny ruch.

Autorzy:
    • Dominik Ludwiński (s26964)
    • Bartosz Dembowski (s29602)
"""

import argparse
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

# Stałe konfiguracji gry
SIZE = 5          # stały rozmiar planszy
DEPTH = 6         # głębokość przeszukiwania algorytmu AI


def attacks(a, b):
    """Funkcja sprawdzająca, czy hetman z pola 'a' atakuje pole 'b'.

    Hetman atakuje po wierszu, kolumnie lub przekątnej.
    Argumenty:
        a, b – krotki (r, c) z numerami wiersza i kolumny.
    Zwraca:
        True – jeśli hetman z 'a' może zaatakować 'b', inaczej False.
    """
    (r1, c1), (r2, c2) = a, b
    return (r1 == r2) or (c1 == c2) or (abs(r1 - r2) == abs(c1 - c2))


def format_moves(moves):
    """Pomocnicza funkcja – wypisanie listy ruchów."""
    return ", ".join(moves) if moves else "(brak)"


class Human(Human_Player):
    """Klasa reprezentująca gracza-człowieka."""

    def ask_move(self, game):
        """Pyta gracza o ruch w formacie r,c (np. 1,3) i sprawdza jego poprawność."""
        while True:
            legal = game.possible_moves()
            print("Dostępne ruchy:", format_moves(legal))
            try:
                raw = input("Twój ruch (r,c np. 1,3): ").strip()
                r_s, c_s = raw.split(',')
                r, c = int(r_s), int(c_s)
                if not (0 <= r < SIZE and 0 <= c < SIZE):
                    print("❌ Poza planszą. Użyj indeksów od 0 do", SIZE - 1)
                    continue
                mv = f"{r},{c}"
                if mv in legal:
                    return mv
                print("❌ Nielegalny ruch: pole zajęte lub konflikt z przeciwnikiem.")
            except Exception:
                print("❌ Błędny format. Wpisz np. 2,4.")


class ReginaPlacement(TwoPlayerGame):
    """Główna klasa gry Regina Placement.

    Zawiera logikę gry, reguły oraz komunikację z silnikiem AI (Negamax).
    """

    def __init__(self, players):
        """Inicjalizacja nowej gry."""
        self.players = players
        self.current_player = 1  # gracz 1 zaczyna
        self.queens = []         # lista hetmanów w postaci (r, c, owner)

    def _occupied(self):
        """Zwraca zbiór wszystkich zajętych pól."""
        return {(r, c) for (r, c, _) in self.queens}

    def _legal_place(self, r, c, owner):
        """Sprawdza, czy można postawić hetmana danego gracza na polu (r, c)."""
        if (r, c) in self._occupied():
            return False
        for (qr, qc, qown) in self.queens:
            if qown != owner and attacks((qr, qc), (r, c)):
                return False
        return True

    def possible_moves(self):
        """Zwraca listę wszystkich możliwych ruchów w postaci 'r,c'."""
        owner = self.current_player
        moves = []
        for r in range(SIZE):
            for c in range(SIZE):
                if self._legal_place(r, c, owner):
                    moves.append(f"{r},{c}")
        return moves

    def make_move(self, move):
        """Wykonuje ruch: dodaje nowego hetmana na planszę."""
        r_s, c_s = move.split(',')
        r, c = int(r_s), int(c_s)
        self.queens.append((r, c, self.current_player))

    def unmake_move(self, move):
        """Cofa ostatni ruch (używane przez AI przy analizie)."""
        self.queens.pop()

    def lose(self):
        """Zwraca True, jeśli gracz na ruchu nie ma żadnego legalnego ruchu."""
        return self.possible_moves() == []

    def is_over(self):
        """Sprawdza, czy gra się zakończyła (czy ktoś przegrał)."""
        return self.lose()

    def scoring(self):
        """Ocena stanu gry z perspektywy gracza na ruchu.
        -100 oznacza przegraną (brak ruchu), 0 w przeciwnym razie.
        """
        return -100 if self.lose() else 0

    def show(self):
        """Wypisuje aktualną planszę w konsoli."""
        board = [[" . " for _ in range(SIZE)] for _ in range(SIZE)]
        for (r, c, own) in self.queens:
            board[r][c] = " A " if own == 1 else " B "
        header = "   " + " ".join(f"{c:2d}" for c in range(SIZE))
        print("\n" + header)
        for r in range(SIZE):
            print(f"{r:2d} " + "".join(board[r]))


def main():
    """Punkt startowy gry.
    Domyślnie gra Człowiek vs AI.
    Flaga --aivai pozwala uruchomić tryb AI kontra AI.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--aivai', action='store_true', help='Pokaz AI vs AI (zamiast Człowiek vs AI)')
    args = p.parse_args()

    ai = Negamax(DEPTH)  # wybór algorytmu dla AI

    if args.aivai:
        game = ReginaPlacement([AI_Player(ai), AI_Player(ai)])
    else:
        game = ReginaPlacement([Human(), AI_Player(ai)])

    print(f"Regina Placement – plansza {SIZE}x{SIZE}")
    print("Zasada: własne hetmany mogą się widzieć (blokujemy tylko A–B)")
    game.play()
    print("Gracz %d przegrywa" % game.current_player)


if __name__ == '__main__':
    main()
