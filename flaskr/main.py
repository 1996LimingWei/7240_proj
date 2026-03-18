"""
Movie Recommender System - Optimized Recommendation Engine

ACTIVE ALGORITHMS (Selected via Offline Evaluation):
1. User-Based CF (Pearson similarity) - Best ranking metrics
2. SVD Matrix Factorization - Best accuracy (MAE=0.68, RMSE=0.88)
3. TF-IDF Content-Based - Fastest (0.01s), uses movie overviews
4. Hybrid (50% SVD + 30% User-CF + 20% TF-IDF) - Best overall

DEPRECATED ALGORITHMS (Kept for reference/documentation):
- Item-Based CF: Slow (2.67s), lower accuracy (MAE=0.78, RMSE=1.00)
- Time-Decay CF: Minimal improvement in evaluation

Evaluation Metrics: MAE, RMSE, Precision@K, Recall@K, nDCG@K
See evaluate_algorithms.py for complete benchmark results.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import (
    Blueprint, render_template, request
)

from .tools.data_tool import *

from surprise import Reader, SVD, accuracy
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from surprise.model_selection import train_test_split
SURPRISE_AVAILABLE = True


bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()

# Pre-compute TF-IDF matrix for content-based recommendations
tfidf_matrix, tfidf_vectorizer = getTfidfMatrix(movies)
content_sim_matrix = getContentSimilarityMatrix(tfidf_matrix)

# Create movie_id to index mapping for content similarity
movie_id_to_idx = {mid: idx for idx,
                   mid in enumerate(movies['movieId'].values)}
idx_to_movie_id = {idx: mid for idx,
                   mid in enumerate(movies['movieId'].values)}

# Pre-compute user-item matrix for collaborative filtering
user_item_matrix = rates.pivot(
    index='userId', columns='movieId', values='rating').fillna(0)
# Ensure no NaN or inf values
user_item_matrix = user_item_matrix.replace([np.inf, -np.inf], 0).fillna(0)
user_ids = user_item_matrix.index.tolist()
movie_ids = user_item_matrix.columns.tolist()
user_item_np = user_item_matrix.values


@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        user_genres = []
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        user_rates = []
    user_likes = request.cookies.get('user_likes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []

    # Get selected algorithm from query parameter
    algorithm = request.args.get('algorithm', 'hybrid')

    default_genres_movies = getMoviesByGenres(user_genres)[:10]

    # Get recommendations based on selected algorithm
    # OPTIMIZED: Only keep best performing algorithms based on evaluation
    recommendations_movies = []
    recommendations_message = ""

    if algorithm == 'svd':
        # Best accuracy (MAE: 0.68, RMSE: 0.88)
        recommendations_movies, recommendations_message = getSVDRecommendations(
            user_rates)
    elif algorithm == 'user_cf':
        # Good ranking metrics, competitive with SVD
        recommendations_movies, recommendations_message = getRecommendationBy(
            user_rates)
    elif algorithm == 'tfidf':
        # Fastest (0.01s), uses overviews, good for cold-start
        recommendations_movies, recommendations_message = getTfidfRecommendations(
            user_likes)
    else:  # hybrid (default) - combines SVD + User-CF + TF-IDF
        recommendations_movies, recommendations_message = getOptimizedHybridRecommendations(
            user_rates, user_likes)

    likes_similar_movies, likes_similar_message = getLikedSimilarBy(
        [int(numeric_string) for numeric_string in user_likes])
    likes_movies = getUserLikesBy(user_likes)

    return render_template('index.html',
                           genres=default_genres,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           user_likes=user_likes,
                           default_genres_movies=default_genres_movies,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           algorithm=algorithm,
                           )


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results


def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))


def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres, columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist(
        )
        results = movies[movies['genres'].apply(
            lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# Helper function: Compute Pearson correlation between two vectors


def pearson_correlation(v1, v2):
    """Compute Pearson correlation between two vectors."""
    mask = (v1 != 0) & (v2 != 0)
    if mask.sum() < 2:
        return 0
    v1_masked = v1[mask]
    v2_masked = v2[mask]
    if v1_masked.std() == 0 or v2_masked.std() == 0:
        return 0
    return np.corrcoef(v1_masked, v2_masked)[0, 1]


# User-Based CF with fallback
def getRecommendationBy(user_rates, k=12):
    """User-Based Collaborative Filtering using Pearson similarity."""
    results = []
    if len(user_rates) > 0:
        user_ratings = ratesFromUser(user_rates)
        current_user_id = 611

        # Build extended user-item matrix including current user
        extended_matrix = user_item_matrix.copy()
        for _, row in user_ratings.iterrows():
            if row['movieId'] in extended_matrix.columns:
                extended_matrix.loc[current_user_id,
                                    row['movieId']] = row['rating']

        # Compute similarities with all users
        current_user_vector = extended_matrix.loc[current_user_id].values
        similarities = []
        for uid in extended_matrix.index:
            if uid != current_user_id:
                other_vector = extended_matrix.loc[uid].values
                sim = pearson_correlation(current_user_vector, other_vector)
                if sim > 0:
                    similarities.append((uid, sim))

        # Sort by similarity and get top-k neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = similarities[:20]

        # Predict ratings for unrated movies
        rated_movies = set(user_ratings['movieId'].tolist())
        predictions = []

        for movie_id in movies['movieId'].unique():
            if movie_id not in rated_movies and movie_id in extended_matrix.columns:
                # Weighted average of neighbor ratings
                weighted_sum = 0
                sim_sum = 0
                for neighbor_id, sim in top_neighbors:
                    rating = extended_matrix.loc[neighbor_id, movie_id]
                    if rating > 0:
                        weighted_sum += sim * rating
                        sim_sum += sim

                if sim_sum > 0:
                    pred_rating = weighted_sum / sim_sum
                    predictions.append((movie_id, pred_rating))

        # Sort and get top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [mid for mid, _ in predictions[:k]]
        results = movies[movies['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        return results.to_dict('records'), "Recommended using User-Based CF (Pearson similarity)."
    return results, "No recommendations."


# Modify this function
def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        # Step 1: Representing items with multi-hot vectors
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(
            movies)
        # Step 2: Building user profile
        user_profile = build_user_profile(
            user_likes, item_rep_vector, feature_list)
        # Step 3: Predicting user interest in items
        results = generate_recommendation_results(
            user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."


# Step 1: Representing items with multi-hot vectors
def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)

    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()

    return movies_genre_matrix, movies_with_genres, genre_list

# Step 2: Building user profile


def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(
        movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T

    if normalized:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile
# Step 3: Predicting user preference for items


def generate_recommendation_results(user_profile, item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(
        by=['similarity'], ascending=False)[:k]
    return rec_result


# ============================================================================
# NEW RECOMMENDATION METHODS
# ============================================================================

# 1. TF-IDF Content-Based Recommendation using Movie Overviews
def getTfidfRecommendations(user_likes, k=12):
    """
    Content-based recommendation using TF-IDF on movie overviews.
    Based on course content: Content-based Filtering Methods based on unstructured content
    """
    if len(user_likes) == 0:
        return [], "No recommendations."

    liked_indices = []
    for movie_id in user_likes:
        if int(movie_id) in movie_id_to_idx:
            liked_indices.append(movie_id_to_idx[int(movie_id)])

    if len(liked_indices) == 0:
        return [], "No recommendations."

    liked_similarities = content_sim_matrix[liked_indices].mean(axis=0)

    movie_scores = []
    for idx, score in enumerate(liked_similarities):
        movie_id = idx_to_movie_id[idx]
        if str(movie_id) not in user_likes:
            movie_scores.append((movie_id, score))

    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in movie_scores[:k]]

    results = movies[movies['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        return results.to_dict('records'), "Recommended based on movie content (TF-IDF on overviews)."
    return [], "No recommendations."


# ============================================================================
# DEPRECATED ALGORITHMS (Kept for reference)
# ============================================================================

# 2. Item-Based Collaborative Filtering [DEPRECATED - Slow (2.67s), Lower Accuracy]
# Evaluation Results: MAE=0.78, RMSE=1.00, Precision@10=0.43, nDCG@10=0.67
# def getItemBasedCFRecommendations(user_rates, k=12):
#     """
#     Item-based collaborative filtering using cosine similarity.
#     Based on course content: k-NN based methods (memory-based) Collaborative Filtering - Item-based
#     """
#     if len(user_rates) == 0:
#         return [], "No recommendations."
#
#     user_ratings = ratesFromUser(user_rates)
#     if len(user_ratings) == 0:
#         return [], "No recommendations."
#
#     rated_movies = user_ratings['movieId'].tolist()
#     user_ratings_dict = dict(
#         zip(user_ratings['movieId'], user_ratings['rating']))
#
#     # Compute item-item cosine similarity matrix
#     item_matrix = user_item_matrix.T.values  # Items as rows
#     item_sim_matrix = cosine_similarity(item_matrix)
#     movie_id_to_col = {mid: i for i,
#                        mid in enumerate(user_item_matrix.columns)}
#
#     # Predict ratings for all movies
#     predictions = []
#     for movie_id in movies['movieId'].unique():
#         if movie_id not in rated_movies and movie_id in movie_id_to_col:
#             col_idx = movie_id_to_col[movie_id]
#             weighted_sum = 0
#             sim_sum = 0
#
#             for rated_movie, rating in user_ratings_dict.items():
#                 if rated_movie in movie_id_to_col:
#                     rated_col = movie_id_to_col[rated_movie]
#                     sim = item_sim_matrix[col_idx, rated_col]
#                     if sim > 0:
#                         weighted_sum += sim * rating
#                         sim_sum += sim
#
#             if sim_sum > 0:
#                 pred_rating = weighted_sum / sim_sum
#                 predictions.append((movie_id, pred_rating))
#
#     predictions.sort(key=lambda x: x[1], reverse=True)
#     top_movie_ids = [mid for mid, _ in predictions[:k]]
#
#     results = movies[movies['movieId'].isin(top_movie_ids)]
#
#     if len(results) > 0:
#         return results.to_dict('records'), "Recommended using Item-Based Collaborative Filtering (cosine similarity)."
#     return [], "No recommendations."


# 3. SVD Matrix Factorization
def getSVDRecommendations(user_rates, k=12):
    """
    Matrix Factorization using SVD (using sklearn's TruncatedSVD).
    Based on course content: Model-based CF - Matrix Factorization (MF)
    """
    if len(user_rates) == 0:
        return [], "No recommendations."

    user_ratings = ratesFromUser(user_rates)
    if len(user_ratings) == 0:
        return [], "No recommendations."

    # Build extended matrix with current user
    extended_matrix = user_item_matrix.copy()
    for _, row in user_ratings.iterrows():
        if row['movieId'] in extended_matrix.columns:
            extended_matrix.loc[611, row['movieId']] = row['rating']

    # Ensure no NaN values before SVD
    extended_matrix = extended_matrix.replace([np.inf, -np.inf], 0).fillna(0)

    # Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(extended_matrix.values)
    item_factors = svd.components_.T

    # Predict ratings
    current_user_idx = extended_matrix.index.get_loc(611)
    user_vector = user_factors[current_user_idx]

    rated_movies = set(user_ratings['movieId'].tolist())
    predictions = []

    for i, movie_id in enumerate(extended_matrix.columns):
        if movie_id not in rated_movies:
            pred_rating = np.dot(user_vector, item_factors[i])
            # Scale to 1-5 range
            pred_rating = np.clip(pred_rating, 1, 5)
            predictions.append((movie_id, pred_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in predictions[:k]]

    results = movies[movies['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        return results.to_dict('records'), "Recommended using SVD Matrix Factorization."
    return [], "No recommendations."


# 4. Time-Decay Weighted Collaborative Filtering [DEPRECATED - Minimal improvement in evaluation]
# Note: Implementation kept to demonstrate temporal dynamics concept from course
# def getTimeDecayRecommendations(user_rates, k=12, decay_factor=0.5):
#     """
#     User-based CF with time-decay weighting.
#     Based on course content: Leveraging timestamps for temporal dynamics
#     """
#     if len(user_rates) == 0:
#         return [], "No recommendations."
#
#     user_ratings = ratesFromUser(user_rates)
#     current_user_id = 611
#
#     # Build extended matrix with time-decay weights
#     extended_matrix = user_item_matrix.copy()
#     time_weights = {}
#
#     for _, row in user_ratings.iterrows():
#         if row['movieId'] in extended_matrix.columns:
#             extended_matrix.loc[current_user_id,
#                                 row['movieId']] = row['rating']
#             # Current ratings have max weight
#             time_weights[row['movieId']] = 1.0
#
#     # Compute similarities
#     current_user_vector = extended_matrix.loc[current_user_id].values
#     similarities = []
#
#     for uid in extended_matrix.index:
#         if uid != current_user_id:
#             other_vector = extended_matrix.loc[uid].values
#             sim = pearson_correlation(current_user_vector, other_vector)
#             if sim > 0:
#                 similarities.append((uid, sim))
#
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     top_neighbors = similarities[:20]
#
#     # Predict with time-decay weighting
#     rated_movies = set(user_ratings['movieId'].tolist())
#     predictions = []
#
#     for movie_id in movies['movieId'].unique():
#         if movie_id not in rated_movies and movie_id in extended_matrix.columns:
#             weighted_sum = 0
#             sim_sum = 0
#
#             for neighbor_id, sim in top_neighbors:
#                 rating = extended_matrix.loc[neighbor_id, movie_id]
#                 if rating > 0:
#                     # Apply time-decay weight if available
#                     weight = time_weights.get(movie_id, 0.5)
#                     weighted_sum += sim * rating * weight
#                     sim_sum += sim * weight
#
#             if sim_sum > 0:
#                 pred_rating = weighted_sum / sim_sum
#                 predictions.append((movie_id, pred_rating))
#
#     predictions.sort(key=lambda x: x[1], reverse=True)
#     top_movie_ids = [mid for mid, _ in predictions[:k]]
#
#     results = movies[movies['movieId'].isin(top_movie_ids)]
#
#     if len(results) > 0:
#         return results.to_dict('records'), f"Recommended using Time-Decay Weighted CF."
#     return [], "No recommendations."


# 5. OPTIMIZED Hybrid Recommendation (Top 3 Methods Only)
def getOptimizedHybridRecommendations(user_rates, user_likes, k=12):
    """
    OPTIMIZED Hybrid - Combines best performing algorithms based on evaluation:
    - SVD (50%) - Best accuracy (MAE: 0.68, RMSE: 0.88)
    - User-based CF (30%) - Good ranking metrics
    - TF-IDF Content-based (20%) - Fast, uses overviews

    REMOVED: Item-Based CF (slow, lower accuracy) and Time-Decay (minimal improvement)
    """
    if len(user_rates) == 0 and len(user_likes) == 0:
        return [], "No recommendations."

    scores = {}

    # 1. SVD scores (50%) - Best performing algorithm
    if len(user_rates) > 0:
        user_ratings = ratesFromUser(user_rates)
        extended_matrix = user_item_matrix.copy()
        for _, row in user_ratings.iterrows():
            if row['movieId'] in extended_matrix.columns:
                extended_matrix.loc[611, row['movieId']] = row['rating']

        # Ensure no NaN values before SVD
        extended_matrix = extended_matrix.replace(
            [np.inf, -np.inf], 0).fillna(0)

        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(extended_matrix.values)
        item_factors = svd.components_.T

        current_user_idx = extended_matrix.index.get_loc(611)
        user_vector = user_factors[current_user_idx]

        rated_movies = set(user_ratings['movieId'].tolist())
        for i, movie_id in enumerate(extended_matrix.columns):
            if movie_id not in rated_movies:
                pred = np.dot(user_vector, item_factors[i])
                pred = np.clip(pred, 1, 5)
                scores[movie_id] = scores.get(movie_id, 0) + 0.5 * pred

    # 2. User-based CF scores (30%) - Competitive ranking metrics
    if len(user_rates) > 0:
        user_ratings = ratesFromUser(user_rates)
        current_user_id = 611

        extended_matrix = user_item_matrix.copy()
        for _, row in user_ratings.iterrows():
            if row['movieId'] in extended_matrix.columns:
                extended_matrix.loc[current_user_id,
                                    row['movieId']] = row['rating']

        current_user_vector = extended_matrix.loc[current_user_id].values
        similarities = []
        for uid in extended_matrix.index:
            if uid != current_user_id:
                other_vector = extended_matrix.loc[uid].values
                sim = pearson_correlation(current_user_vector, other_vector)
                if sim > 0:
                    similarities.append((uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = similarities[:20]

        rated_movies = set(user_ratings['movieId'].tolist())
        for movie_id in movies['movieId'].unique():
            if movie_id not in rated_movies and movie_id in extended_matrix.columns:
                weighted_sum = 0
                sim_sum = 0
                for neighbor_id, sim in top_neighbors:
                    rating = extended_matrix.loc[neighbor_id, movie_id]
                    if rating > 0:
                        weighted_sum += sim * rating
                        sim_sum += sim
                if sim_sum > 0:
                    scores[movie_id] = scores.get(
                        movie_id, 0) + 0.3 * (weighted_sum / sim_sum)

    # 3. TF-IDF Content scores (20%) - Fast and uses overviews
    if len(user_likes) > 0:
        liked_indices = [movie_id_to_idx[int(mid)] for mid in user_likes if int(
            mid) in movie_id_to_idx]
        if liked_indices:
            content_scores = content_sim_matrix[liked_indices].mean(axis=0)
            for idx, score in enumerate(content_scores):
                movie_id = idx_to_movie_id[idx]
                normalized_score = 1 + 4 * score
                scores[movie_id] = scores.get(
                    movie_id, 0) + 0.2 * normalized_score

    # Sort and get top-k
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in sorted_scores[:k]]

    results = movies[movies['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        return results.to_dict('records'), "Recommended using Optimized Hybrid (SVD 50% + User-CF 30% + TF-IDF 20%)."
    return [], "No recommendations."
