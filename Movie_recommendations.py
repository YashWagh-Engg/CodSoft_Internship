# ---------------- MOVIE DATASET ----------------
movies = [
    {"title": "Avengers", "genre": ["Action", "Sci-Fi"]},
    {"title": "John Wick", "genre": ["Action", "Thriller"]},
    {"title": "Inception", "genre": ["Sci-Fi", "Thriller"]},
    {"title": "Interstellar", "genre": ["Sci-Fi", "Drama"]},
    {"title": "Titanic", "genre": ["Romance", "Drama"]},
    {"title": "The Notebook", "genre": ["Romance", "Drama"]},
    {"title": "The Dark Knight", "genre": ["Action", "Crime"]},
    {"title": "Gladiator", "genre": ["Action", "Drama"]},
    {"title": "Forrest Gump", "genre": ["Drama", "Comedy"]},
    {"title": "The Hangover", "genre": ["Comedy"]},
    {"title": "Jumanji", "genre": ["Adventure", "Comedy"]},
    {"title": "Jurassic Park", "genre": ["Adventure", "Sci-Fi"]},
    {"title": "Shutter Island", "genre": ["Thriller", "Mystery"]},
    {"title": "Parasite", "genre": ["Thriller", "Drama"]},
    {"title": "La La Land", "genre": ["Romance", "Music"]},
    {"title": "The Conjuring", "genre": ["Horror", "Mystery"]},
    {"title": "It", "genre": ["Horror"]},
    {"title": "Toy Story", "genre": ["Animation", "Comedy"]},
    {"title": "Coco", "genre": ["Animation", "Music"]},
    {"title": "Up", "genre": ["Animation", "Adventure"]}
]

# ---------------- SHOW GENRES FUNCTION ----------------
def show_genres(movies):
    genres = set()
    for movie in movies:
        for g in movie["genre"]:
            genres.add(g)
    return sorted(genres)

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_movies(movies, user_genres):
    recommendations = []

    for movie in movies:
        match_count = len(set(movie["genre"]).intersection(user_genres))
        if match_count > 0:
            recommendations.append((movie["title"], match_count))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# ---------------- MAIN PROGRAM ----------------
print("Welcome to Movie Recommendation System\n")

available_genres = show_genres(movies)
print("Available Genres:")
print(", ".join(available_genres))

user_input = input("\nEnter your preferred genres (comma-separated): ")
user_genres = [genre.strip().title() for genre in user_input.split(",")]

results = recommend_movies(movies, user_genres)

print("\nRecommended Movies for You:\n")

if results:
    for movie, score in results:
        print(f"- {movie} (Matched Genres: {score})")
else:
    print("No movies found matching your preferences.")