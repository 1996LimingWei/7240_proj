import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/movie_info.csv"
    df = pd.read_csv(path)
    df['genres'] = df.genres.str.split('|')
    # Fill NaN overviews with empty string
    df['overview'] = df['overview'].fillna('')
    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/genre.csv"
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index('id')
    return df


# user id, item id, rating, timestamp
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/ratings.csv"
    df = pd.read_csv(path, delimiter=",", header=0, names=[
                     "userId", "movieId", "rating", "timestamp"])
    # Keep timestamp for temporal analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df


# Get ratings with time-decay weights
def getRatesWithTimeDecay(decay_factor=0.5):
    """
    Get ratings with time-decay weights.
    More recent ratings get higher weights.
    decay_factor: controls how fast weights decay (0-1, higher = faster decay)
    """
    df = getRates()
    max_time = df['timestamp'].max()
    min_time = df['timestamp'].min()
    time_range = (max_time - min_time).total_seconds()

    # Calculate time decay weights
    df['time_weight'] = df['timestamp'].apply(
        lambda x: np.exp(-decay_factor * (max_time -
                         x).total_seconds() / time_range)
        if time_range > 0 else 1.0
    )
    return df


# Compute TF-IDF matrix for movie overviews
def getTfidfMatrix(movies_df, max_features=5000):
    """
    Compute TF-IDF matrix for movie overviews.
    Returns the TF-IDF matrix and the vectorizer.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'])
    return tfidf_matrix, tfidf


# Compute content similarity matrix
def getContentSimilarityMatrix(tfidf_matrix):
    """
    Compute cosine similarity matrix from TF-IDF matrix.
    """
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)


# ============================================================================
# EVALUATION METRICS
# Based on course content: Rating prediction accuracy and ranking metrics
# ============================================================================

def evaluate_rating_prediction(y_true, y_pred):
    """
    Evaluate rating prediction accuracy.
    Metrics: MAE, RMSE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4)}


def precision_at_k(recommended_items, relevant_items, k):
    """Calculate Precision@K."""
    if k == 0:
        return 0.0
    recommended_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    n_relevant_and_recommended = len(recommended_k & relevant_set)
    return n_relevant_and_recommended / k


def recall_at_k(recommended_items, relevant_items, k):
    """Calculate Recall@K."""
    if len(relevant_items) == 0:
        return 0.0
    recommended_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    n_relevant_and_recommended = len(recommended_k & relevant_set)
    return n_relevant_and_recommended / len(relevant_set)


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """Calculate nDCG@K."""
    relevance_scores = {item: 1 for item in relevant_items}
    relevances = [relevance_scores.get(item, 0)
                  for item in recommended_items[:k]]

    # DCG
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    dcg = np.sum(np.array(relevances) / discounts)

    # IDCG
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    if len(ideal_relevances) == 0:
        return 0.0
    idcg = np.sum(np.array(ideal_relevances) /
                  np.log2(np.arange(2, len(ideal_relevances) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking_metrics(test_ratings, recommendations, k_values=[5, 10, 20]):
    """
    Evaluate ranking metrics for recommendations.
    Returns Precision@K, Recall@K, nDCG@K for each K.
    """
    results = {k: {'precision': [], 'recall': [], 'ndcg': []}
               for k in k_values}

    user_ratings = test_ratings.groupby('userId')

    for user_id, group in user_ratings:
        if user_id not in recommendations:
            continue

        relevant_items = set(group[group['rating'] >= 4]['movieId'].tolist())
        recommended_items = recommendations[user_id]

        for k in k_values:
            precision_k = precision_at_k(recommended_items, relevant_items, k)
            recall_k = recall_at_k(recommended_items, relevant_items, k)
            ndcg_k = ndcg_at_k(recommended_items, relevant_items, k)

            results[k]['precision'].append(precision_k)
            results[k]['recall'].append(recall_k)
            results[k]['ndcg'].append(ndcg_k)

    avg_results = {}
    for k in k_values:
        avg_results[f'Precision@{k}'] = round(
            np.mean(results[k]['precision']), 4)
        avg_results[f'Recall@{k}'] = round(np.mean(results[k]['recall']), 4)
        avg_results[f'nDCG@{k}'] = round(np.mean(results[k]['ndcg']), 4)

    return avg_results
