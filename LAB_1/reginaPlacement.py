"""
    Regina Placement – gra logiczna oparta o easyAI (TwoPlayerGame + Negamax)
    ========================================================================
    Gra polega na naprzemiennym stawianiu hetmanów (figur szachowych Queen)
    na planszy 5x5. Celem gry jest strategiczne rozmieszczanie hetmanów tak,
    aby zablokować przeciwnika i uniemożliwić mu wykonanie dalszego ruchu.

    Zasady gry:
     - Gracze (A i B) na przemian stawiają swoje hetmany na planszy.
     - Ruch wykonuje się poprzez podanie współrzędnych w formacie r,c (np. 1,3).
     - Hetman atakuje w wierszu, kolumnie i po przekątnych.
     - Nie można postawić hetmana na polu, które jest już zajęte lub atakowane
       przez hetmana przeciwnika.
     - Własne hetmany mogą się "widzieć" – nie blokują się nawzajem.
     - Gracz, który nie może wykonać ruchu, przegrywa.

    Jak przygotować środowisko:
     - Zainstalować easyAI: pip install easyAI
     - Uruchomienie gry:
         python regina_placement_simple.py          # Człowiek vs AI
         python regina_placement_simple.py --aivai  # AI vs AI (pokaz)

    Autorzy:
     - Dominik Ludwiński (s26964)
     - Bartosz Dembowski (s29602)
"""

import argparse
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

# --- Stałe gry ---
SIZE = 5          # rozmiar planszy
DEPTH = 6         # głębokość przeszukiwania AI


def attacks(a, b):
    """Funkcja sprawdzająca, czy hetman z pola 'a' atakuje pole 'b'.

    Parametry:
     - a (tuple): współrzędne pierwszego pola (wiersz, kolumna)
     - b (tuple): współrzędne drugiego pola (wiersz, kolumna)

    Zwraca:
     - bool: True, jeśli hetman z pola 'a' może zaatakować pole 'b'
    """
    (r1, c1), (r2, c2) = a, b
    return (r1 == r2) or (c1 == c2) or (abs(r1 - r2) == abs(c1 - c2))


def format_moves(moves):
    """Funkcja formatująca listę dostępnych ruchów do wypisania w konsoli.

    Parametry:
     - moves (list): lista możliwych ruchów

    Zwraca:
     - string: sformatowany tekst z ruchami
    """
    return ", ".join(moves) if moves else "(brak)"


class Human(Human_Player):
    """Klasa reprezentująca gracza-człowieka."""

    def ask_move(self, game):
        """Funkcja prosząca użytkownika o wykonanie ruchu.

        Parametry:
         - game (ReginaPlacement): aktualna instancja gry

        Zwraca:
         - string: ruch w formacie 'r,c' (np. '1,3')
        """
        while True:
            legal = game.possible_moves()
            print("Dostępne ruchy:", format_moves(legal))
            try:
                raw = input("Twój ruch (r,c np. 1,3): ").strip()
                r_s, c_s = raw.split(',')
                r, c = int(r_s), int(c_s)
                if not (0 <= r < SIZE and 0 <= c < SIZE):
                    print("❌ Poza planszą. Indeksy od 0 do", SIZE - 1)
                    continue
                mv = f"{r},{c}"
                if mv in legal:
                    return mv
                print("❌ Nielegalny ruch: pole zajęte lub konflikt z przeciwnikiem.")
            except Exception:
                print("❌ Błędny format. Wpisz np. 2,4.")


class ReginaPlacement(TwoPlayerGame):
    """Główna klasa gry 'Regina Placement'.

    Dziedziczy po easyAI.TwoPlayerGame.
    Implementuje logikę gry oraz współpracuje z algorytmem Negamax.
    """

    def __init__(self, players):
        """Funkcja inicjalizująca nową grę.

        Parametry:
         - players (list): lista dwóch graczy [Player1, Player2]
        """
        self.players = players
        self.current_player = 1  # gracz 1 zaczyna
        self.queens = []         # lista hetmanów w formacie (r, c, owner)

    def _occupied(self):
        """Funkcja zwracająca zbiór zajętych pól.

        Zwraca:
         - set: zbiór krotek (r, c)
        """
        return {(r, c) for (r, c, _) in self.queens}

    def _legal_place(self, r, c, owner):
        """Sprawdza, czy można postawić hetmana danego gracza na polu (r, c).

        Parametry:
         - r (int): numer wiersza
         - c (int): numer kolumny
         - owner (int): numer gracza (1 lub 2)

        Zwraca:
         - bool: True, jeśli ruch jest dozwolony
        """
        if (r, c) in self._occupied():
            return False
        for (qr, qc, qown) in self.queens:
            if qown != owner and attacks((qr, qc), (r, c)):
                return False
        return True

    def possible_moves(self):
        """Funkcja zwraca wszystkie możliwe ruchy w danym momencie gry.

        Zwraca:
         - list: lista ruchów w formacie 'r,c'
        """
        owner = self.current_player
        moves = []
        for r in range(SIZE):
            for c in range(SIZE):
                if self._legal_place(r, c, owner):
                    moves.append(f"{r},{c}")
        return moves

    def make_move(self, move):
        """Funkcja wykonująca ruch – dodaje hetmana do listy figur.

        Parametry:
         - move (string): ruch w formacie 'r,c'
        """
        r_s, c_s = move.split(',')
        r, c = int(r_s), int(c_s)
        self.queens.append((r, c, self.current_player))

    def unmake_move(self, move):
        """Funkcja cofająca ostatni ruch (używana przez AI podczas symulacji).

        Parametry:
         - move (string): ruch do cofnięcia
        """
        self.queens.pop()

    def lose(self):
        """Funkcja określająca, czy gracz na ruchu przegrał.

        Zwraca:
         - bool: True, jeśli gracz nie ma legalnych ruchów
        """
        return self.possible_moves() == []

    def scoring(self):
        """Funkcja oceniająca aktualny stan gry dla AI.

        Zwraca:
         - int: -100 dla przegranego stanu, 0 w przeciwnym razie
        """
        return -100 if self.lose() else 0

    def is_over(self):
        """Funkcja sprawdzająca, czy gra się zakończyła.

        Zwraca:
         - bool: True, jeśli gra jest skończona
        """
        return self.lose()

    def show(self):
        """Funkcja wypisująca aktualny stan planszy w konsoli.

        Symbole:
         - 'A' – hetman gracza 1
         - 'B' – hetman gracza 2
         - '.' – puste pole
        """
        board = [[" . " for _ in range(SIZE)] for _ in range(SIZE)]
        for (r, c, own) in self.queens:
            board[r][c] = " A " if own == 1 else " B "
        header = "   " + " ".join(f"{c:2d}" for c in range(SIZE))
        print("\n" + header)
        for r in range(SIZE):
            print(f"{r:2d} " + "".join(board[r]))


def main():
    """Funkcja główna uruchamiająca grę.

    Flagi:
     - --aivai : uruchamia tryb AI kontra AI (pokaz)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--aivai', action='store_true', help='Pokaz AI vs AI (zamiast Człowiek vs AI)')
    args = parser.parse_args()

    ai = Negamax(DEPTH)  # algorytm przeszukiwania

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