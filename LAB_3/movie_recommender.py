"""
Nazwa: Silnik rekomendacji filmów / seriali

Autorzy:
Dominik Ludwiński
Bartosz Dembowski

Program implementuje prosty silnik rekomendacji filmów na podstawie ocen użytkowników
oraz klasteryzacji użytkowników (k-means).

Dla wskazanego użytkownika:
- znajduje 5 filmów, których nie widział, a które mogą go zainteresować (rekomendacje),
- znajduje 5 filmów, których nie powinien oglądać (antyrekomendacje),
- dociąga dodatkowe informacje z zewnętrznego API OMDb (tytuł, rok, gatunek, ocena IMDb).

Wymagania:
  - Python 3.x
  - Biblioteki:
      pip install numpy scikit-learn requests
  - Plik JSON z ocenami w tym samym folderze:
      movie_ratings.json

Instrukcja użycia (przykład):
  python movie_recommender.py --user "Anna Nowak"
"""

import argparse
import json
import os

import numpy as np
import requests
from sklearn.cluster import KMeans


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RATINGS_FILE = os.path.join(SCRIPT_DIR, "movie_ratings.json")

# klucz API do OMDb
OMDB_API_KEY = 'bc93a3bb'


def build_arg_parser():
    """
    Buduje parser argumentów programu.

    Obsługiwany argument:
      --user  - nazwa użytkownika, dla którego liczymy rekomendacje.
    """
    parser = argparse.ArgumentParser(
        description="Silnik rekomendacji filmów z ocen i klasteryzacji k-means."
    )
    parser.add_argument(
        "--user",
        dest="user",
        required=True,
        help="Użytkownik, dla którego zostaną wygenerowane rekomendacje."
    )
    return parser


def load_ratings():
    """
    Wczytuje oceny filmów z pliku movie_ratings.json.

    Struktura:
      {
        "Uzytkownik1": {"Film A": 5, "Film B": 3},
        "Uzytkownik2": {"Film A": 4, "Film C": 2}
      }
    """
    with open(RATINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_user_item_matrix(ratings):
    """
    Buduje macierz użytkownik–film na podstawie ocen.

    Zwraca:
      - macierz numpy [liczba_użytkowników x liczba_filmów]
      - listę użytkowników
      - listę filmów
    """
    users = list(ratings.keys())

    movies_set = set()
    for user_ratings in ratings.values():
        movies_set.update(user_ratings.keys())
    movies = sorted(list(movies_set))

    matrix = np.zeros((len(users), len(movies)), dtype=float)

    for i, user in enumerate(users):
        for j, movie in enumerate(movies):
            if movie in ratings[user]:
                matrix[i, j] = ratings[user][movie]

    return matrix, users, movies


def cluster_users_kmeans(user_item_matrix, n_clusters=3, random_state=42):
    """
    Klasteryzuje użytkowników algorytmem k-means.

    Zwraca wektor etykiet klastrów dla każdego użytkownika.
    """
    n_users = user_item_matrix.shape[0]
    k = min(n_clusters, n_users)
    if k <= 1:
        return np.zeros(n_users, dtype=int)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(user_item_matrix)
    return labels


def aggregate_movie_scores_for_user(ratings, target_user, allowed_users):
    """
    Liczy średnią ocenę filmów, których target_user nie oglądał,
    biorąc pod uwagę tylko użytkowników z tego samego klastra.

    Zwraca słownik:
      film -> średnia_ocena_w_klastrze
    """
    target_movies = ratings[target_user].keys()
    movie_sum = {}
    movie_count = {}

    for user in allowed_users:
        if user == target_user:
            continue
        user_ratings = ratings.get(user, {})
        for movie, score in user_ratings.items():
            if movie in target_movies:
                continue
            movie_sum[movie] = movie_sum.get(movie, 0.0) + score
            movie_count[movie] = movie_count.get(movie, 0) + 1

    average_scores = {}
    for movie in movie_sum:
        count = movie_count[movie]
        if count > 0:
            average_scores[movie] = movie_sum[movie] / count

    return average_scores


def get_top_and_bottom_movies(movie_scores, top_n=5):
    """
    Zwraca:
      - listę top_n najlepszych filmów
      - listę top_n najgorszych filmów
    na podstawie średnich ocen.
    """
    if not movie_scores:
        return [], []

    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_movies[:top_n]
    bottom = sorted(movie_scores.items(), key=lambda x: x[1])[:top_n]
    return top, bottom


def fetch_movie_info_omdb(title):
    """
    Pobiera podstawowe informacje o filmie z OMDb.

    Zwraca słownik z polami m.in.:
      Title, Year, Genre, imdbRating
    albo pusty słownik, jeśli nie uda się pobrać.
    """
    if not OMDB_API_KEY:
        return {}

    params = {
        "t": title,
        "apikey": OMDB_API_KEY,
        "type": "movie"
    }
    try:
        response = requests.get("https://www.omdbapi.com/", params=params, timeout=5)
        data = response.json()
        if data.get("Response") == "False":
            return {}
        return data
    except Exception:
        return {}


def print_movie_list(header, movies):
    """
    Drukuje listę filmów z nagłówkiem i podstawowymi danymi z OMDb.
    """
    print("\n" + header)
    print("-" * len(header))

    if not movies:
        print("Brak filmów.")
        return

    for title, score in movies:
        info = fetch_movie_info_omdb(title)

        if not info:
            print(f"\nTytuł: {title}")
            print(f"  Średnia ocena w klastrze: {score:.2f}")
            print(f"  Dodatkowe informacje: brak w bazie OMDb")
            continue

        year = info.get("Year", "brak danych")
        genre = info.get("Genre", "brak danych")
        imdb_rating = info.get("imdbRating", "brak danych")

        print(f"\nTytuł: {title}")
        print(f"  Średnia ocena w klastrze: {score:.2f}")
        print(f"  Rok: {year}")
        print(f"  Gatunek: {genre}")
        print(f"  Ocena IMDb: {imdb_rating}")


def main():
    """
    Główna funkcja programu.

    Kroki:
      1. Wczytanie ocen.
      2. Klasteryzacja użytkowników (k-means).
      3. Wypisanie wszystkich klastrów z użytkownikami.
      4. Wybranie klastra danego użytkownika.
      5. Liczenie średnich ocen filmów w tym klastrze.
      6. Wypisanie 5 rekomendacji i 5 antyrekomendacji.
    """
    args = build_arg_parser().parse_args()
    user = args.user

    ratings = load_ratings()

    # sprawdzenie czy istnieje user
    if user not in ratings:
        print(f"Użytkownik '{user}' nie istnieje w bazie ocen.")
        return

    user_item_matrix, users, movies = build_user_item_matrix(ratings)
    labels = cluster_users_kmeans(user_item_matrix, n_clusters=3)

    # wypisanie wszystkich klastrów z użytkownikami
    clusters = {}
    for username, label in zip(users, labels):
        clusters.setdefault(label, []).append(username)

    print("Klastery użytkowników:")
    for cluster_id, members in clusters.items():
        print(f"  Klaster {cluster_id}: {', '.join(members)}")

    # klaster wskazanego użytkownika
    user_index = users.index(user)
    user_cluster = labels[user_index]
    allowed_users = [u for u, label in zip(users, labels) if label == user_cluster]

    # liczymy średnie oceny filmów w klastrze
    movie_scores = aggregate_movie_scores_for_user(ratings, user, allowed_users)
    top_movies, bottom_movies = get_top_and_bottom_movies(movie_scores, top_n=5)

    # wypisujemy wyniki
    print_movie_list(
        header=f"5 filmów rekomendowanych dla użytkownika {user}",
        movies=top_movies
    )
    print_movie_list(
        header=f"5 filmów odradzanych (antyrekomendacje) dla użytkownika {user}",
        movies=bottom_movies
    )


if __name__ == "__main__":
    main()
