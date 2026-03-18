"""
Comprehensive Offline Evaluation of Recommendation Algorithms
"""

import time
from surprise.model_selection import train_test_split
from surprise import Reader, SVD, KNNBasic, KNNWithMeans, Dataset, accuracy
from flaskr.main import (
    getRecommendationBy, getItemBasedCFRecommendations, getSVDRecommendations,
    getTimeDecayRecommendations, getTfidfRecommendations, getOptimizedHybridRecommendations,
    pearson_correlation, user_item_matrix
)
from flaskr.tools.data_tool import getMovies, getRates, loadData, evaluate_rating_prediction, evaluate_ranking_metrics
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/leo/Desktop/ITM/7240/7240_proj')


# Load data
print("Loading data...")
movies, genres, rates = loadData()
print(f"Movies: {len(movies)}, Ratings: {len(rates)}")


def generate_user_ratings_sample(n_ratings=20, random_state=42):
    """Generate a sample user profile from existing ratings."""
    np.random.seed(random_state)
    sample_user = rates.sample(n=n_ratings)[['movieId', 'rating']]
    user_rates = []
    for _, row in sample_user.iterrows():
        user_rates.append(f"611|{int(row['movieId'])}|{int(row['rating'])}")
    return user_rates, sample_user['movieId'].tolist()


def evaluate_algorithm_with_surprise(algo_name, algo, trainset, testset, k_values=[5, 10, 20]):
    """Evaluate a surprise-based algorithm."""
    print(f"\nEvaluating {algo_name}...")
    start_time = time.time()

    # Train
    algo.fit(trainset)

    # Predictions
    predictions = algo.test(testset)

    # Rating prediction metrics
    y_true = [pred.r_ui for pred in predictions]
    y_pred = [pred.est for pred in predictions]
    rating_metrics = evaluate_rating_prediction(y_true, y_pred)

    # Build recommendations dict for ranking metrics
    from collections import defaultdict
    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((pred.iid, pred.est, pred.r_ui))

    recommendations = {}
    test_data = []
    for user_id, preds in user_predictions.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = [iid for iid, _, _ in preds]
        for iid, est, true_r in preds:
            test_data.append(
                {'userId': user_id, 'movieId': iid, 'rating': true_r})

    test_ratings = pd.DataFrame(test_data)
    ranking_metrics = evaluate_ranking_metrics(
        test_ratings, recommendations, k_values)

    elapsed_time = time.time() - start_time

    results = {
        'Algorithm': algo_name,
        'Time (s)': round(elapsed_time, 2),
        **rating_metrics,
        **ranking_metrics
    }

    return results


def evaluate_custom_algorithms():
    """Evaluate custom implementations (TF-IDF and Hybrid)."""
    print("\n" + "="*80)
    print("Evaluating Custom Algorithms")
    print("="*80)

    # Generate sample user
    user_rates, user_likes = generate_user_ratings_sample(n_ratings=20)

    results = []

    # Test each algorithm
    algorithms = [
        ('TF-IDF Content-Based',
         lambda: getTfidfRecommendations([str(m) for m in user_likes], k=20)),
    ]

    for algo_name, algo_func in algorithms:
        print(f"\nTesting {algo_name}...")
        start_time = time.time()

        recs, message = algo_func()
        elapsed_time = time.time() - start_time

        print(
            f"  Generated {len(recs)} recommendations in {elapsed_time:.2f}s")
        print(f"  Message: {message}")

        results.append({
            'Algorithm': algo_name,
            'Time (s)': round(elapsed_time, 2),
            'Recommendations': len(recs)
        })

    return results


def run_surprise_evaluation():
    """Run evaluation on Surprise-based algorithms."""
    print("\n" + "="*80)
    print("Evaluating Surprise-based Algorithms")
    print("="*80)

    # Prepare data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rates[['userId', 'movieId', 'rating']], reader)

    # Split data
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    results = []

    # 1. User-Based CF
    algo = KNNWithMeans(k=20, sim_options={
                        'name': 'pearson', 'user_based': True}, verbose=False)
    results.append(evaluate_algorithm_with_surprise(
        'User-Based CF (Pearson)', algo, trainset, testset))

    # 2. Item-Based CF
    algo = KNNBasic(k=20, sim_options={
                    'name': 'cosine', 'user_based': False}, verbose=False)
    results.append(evaluate_algorithm_with_surprise(
        'Item-Based CF (Cosine)', algo, trainset, testset))

    # 3. SVD
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005,
               reg_all=0.02, random_state=42, verbose=False)
    results.append(evaluate_algorithm_with_surprise(
        'SVD Matrix Factorization', algo, trainset, testset))

    return results


def compare_all_methods():
    """Compare all recommendation methods."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*80)

    all_results = []

    # Run Surprise-based evaluations
    surprise_results = run_surprise_evaluation()
    all_results.extend(surprise_results)

    # Run custom algorithm evaluations
    custom_results = evaluate_custom_algorithms()

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    if surprise_results:
        df = pd.DataFrame(surprise_results)
        print("\nSurprise-based Algorithms:")
        print(df.to_string(index=False))

    if custom_results:
        df = pd.DataFrame(custom_results)
        print("\nCustom Algorithms:")
        print(df.to_string(index=False))

    # Recommendations based on results
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FINAL SYSTEM")
    print("="*80)
    print("""
Based on typical performance patterns:

1. SVD Matrix Factorization
   - Best overall accuracy (lowest RMSE/MAE)
   - Good ranking metrics
   - Scalable and well-established

2. Item-Based CF
   - Good for explaining recommendations
   - Efficient for movie-movie similarity
   - Works well with sparse data

3. TF-IDF Content-Based
   - Solves cold-start problem
   - Uses movie overviews (unique data source)
   - Good diversity in recommendations

4. HYBRID: Combine top 3 methods
   - Weighted ensemble for best performance
   - Already implemented in getHybridRecommendations()

5. User-Based CF
   - Often outperformed by Item-Based and SVD
   - Computationally expensive
   - Can be replaced by SVD for user similarity

6. Time-Decay
   - May not show significant improvement on this dataset
   - Could be enabled as a feature flag
    """)


if __name__ == "__main__":
    compare_all_methods()
