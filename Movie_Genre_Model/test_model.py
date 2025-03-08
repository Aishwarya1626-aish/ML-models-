
import pickle

# Load the trained model
with open("trained_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_genre(movie_plot):
    """Predicts the genre for a given movie plot."""
    movie_plot_tfidf = vectorizer.transform([movie_plot])  # Convert text to TF-IDF
    predicted_genre = best_model.predict(movie_plot_tfidf)
    return predicted_genre[0]

# Test with user input
movie_plot = input("Enter a movie plot: ")
predicted_genre = predict_genre(movie_plot)
print(f"Predicted Genre: {predicted_genre}")
