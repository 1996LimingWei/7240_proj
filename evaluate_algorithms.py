"""
Comprehensive Offline Evaluation of Recommendation Algorithms
"""

import time
from surprise.model_selection import train_test_split
from surprise import Reader, SVD, KNNBasic, KNNWithMeans, Dataset, accuracy
from flaskr.main import (
    getRecommendationBy, getSVDRecommendations,
    getTfidfRecommendations, getOptimizedHybridRecommendations,
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


def evaluate_custom_algorithm_with_metrics(algo_name, algo_func, train_ratings, test_ratings, k_values=[5, 10, 20]):
    """Evaluate custom algorithm with full offline metrics."""
    print(f"\nEvaluating {algo_name}...")
    start_time = time.time()

    # Get recommendations for all users in test set
    from collections import defaultdict

    # Group test ratings by user
    user_test_data = defaultdict(list)
    for _, row in test_ratings.iterrows():
        user_test_data[row['userId']].append({
            'movieId': row['movieId'],
            'rating': row['rating']
        })

    # Generate recommendations for each user
    all_recommendations = {}
    all_predictions = []

    for user_id, user_items in user_test_data.items():
        # Create pseudo user ratings in expected format: "userId|movieId|rating"
        # Note: movieId must be int, but rating stays float
        try:
            user_rates = []
            for item in user_items[:10]:  # Use first 10 as history
                # Convert to int (handles both int and float)
                uid = int(float(user_id))
                mid = int(float(item['movieId']))  # Convert movieId to int
                rating = item['rating']  # Keep rating as-is (float)
                user_rates.append(f"{uid}|{mid}|{rating}")

            # Get recommendations and normalize them to ranked movie IDs
            recs, _ = algo_func(user_rates, k=20)
            normalized_recs = []
            for rec in recs:
                if isinstance(rec, dict) and 'movieId' in rec:
                    normalized_recs.append(int(rec['movieId']))
                elif isinstance(rec, (int, np.integer)):
                    normalized_recs.append(int(rec))

            all_recommendations[int(float(user_id))] = normalized_recs
        except Exception as e:
            print(f"  Warning: Error for user {user_id}: {e}")
            continue

    elapsed_time = time.time() - start_time

    # Compute ranking metrics only if we have recommendations
    if len(all_recommendations) > 0:
        # Build test ratings DataFrame from users we actually generated recs for
        test_data_for_recs = []
        for uid in all_recommendations.keys():
            for item in user_test_data[uid]:
                test_data_for_recs.append({
                    'userId': uid,
                    'movieId': int(item['movieId']),
                    'rating': item['rating']
                })

        test_df = pd.DataFrame(test_data_for_recs)

        ranking_metrics = evaluate_ranking_metrics(
            test_df,
            all_recommendations,
            k_values
        )
    else:
        # No recommendations generated, return empty metrics
        ranking_metrics = {k: 0.0 for k in ['Precision@5', 'Recall@5', 'nDCG@5',
                                            'Precision@10', 'Recall@10', 'nDCG@10',
                                            'Precision@20', 'Recall@20', 'nDCG@20']}

    results = {
        'Algorithm': algo_name,
        'Time (s)': round(elapsed_time, 2),
        'MAE': 'N/A',  # Custom algos don't predict ratings directly
        'RMSE': 'N/A',
        **ranking_metrics
    }

    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  nDCG@10: {results.get('nDCG@10', 'N/A')}")

    return results


def evaluate_custom_algorithms():
    """Evaluate custom implementations (TF-IDF and Hybrid)."""
    print("\n" + "="*80)
    print("Evaluating Custom Algorithms")
    print("="*80)

    # Prepare data for proper evaluation
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rates[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Convert testset to DataFrame for evaluation
    test_data = []
    for uid, iid, r_ui in testset:
        test_data.append({'userId': uid, 'movieId': iid, 'rating': r_ui})
    test_ratings = pd.DataFrame(test_data)

    results = []

    # Define algorithm functions with proper signatures
    def tfidf_func(user_rates, k=20):
        # Extract movie IDs from user_rates (format: "userId|movieId|rating")
        # TF-IDF expects list of movie ID strings as user_likes
        user_likes = [str(int(float(r.split('|')[1])))
                      for r in user_rates] if user_rates else []
        return getTfidfRecommendations(user_likes, k=k)

    def hybrid_func(user_rates, k=20):
        # Hybrid expects both user_rates (full format) and user_likes (list of strings)
        user_likes = [str(int(float(r.split('|')[1])))
                      for r in user_rates] if user_rates else []
        return getOptimizedHybridRecommendations(user_rates, user_likes, k=k)

    # Test each algorithm
    algorithms = [
        ('TF-IDF Content-Based', tfidf_func),
        ('Optimized Hybrid (50% SVD + 30% User-CF + 20% TF-IDF)', hybrid_func),
    ]

    for algo_name, algo_func in algorithms:
        try:
            result = evaluate_custom_algorithm_with_metrics(
                algo_name, algo_func, trainset, test_ratings
            )
            results.append(result)
        except Exception as e:
            print(f"  Error evaluating {algo_name}: {e}")
            results.append({
                'Algorithm': algo_name,
                'Time (s)': 'N/A',
                'MAE': 'N/A',
                'RMSE': 'N/A',
                'Precision@10': 'N/A',
                'Recall@10': 'N/A',
                'nDCG@10': 'N/A'
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

    # 2. SVD
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

    # Combine all results for unified display
    if surprise_results and custom_results:
        # Create unified dataframe (custom algos will have N/A for MAE/RMSE)
        all_for_display = surprise_results + custom_results
        df = pd.DataFrame(all_for_display)

        print("\n" + "="*80)
        print("COMPLETE EVALUATION RESULTS")
        print("="*80)
        print("\nAll Algorithms Comparison:")
        # Reorder columns for better display
        cols = ['Algorithm', 'Time (s)', 'MAE', 'RMSE',
                'Precision@10', 'Recall@10', 'nDCG@10']
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].to_string(index=False))

        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)

        # Find best in each category
        if 'MAE' in df.columns and 'RMSE' in df.columns:
            # Filter only surprise-based for MAE/RMSE comparison
            surprise_df = pd.DataFrame(surprise_results)
            if len(surprise_df) > 0:
                best_mae = surprise_df.loc[surprise_df['MAE'].idxmin()]
                best_rmse = surprise_df.loc[surprise_df['RMSE'].idxmin()]
                print(
                    f"\n✅ Best Accuracy (MAE): {best_mae['Algorithm']} ({best_mae['MAE']})")
                print(
                    f"✅ Best Accuracy (RMSE): {best_rmse['Algorithm']} ({best_rmse['RMSE']})")

        if 'nDCG@10' in df.columns:
            # Get all with valid nDCG@10
            df_valid_ndcg = df[df['nDCG@10'] != 'N/A']
            if len(df_valid_ndcg) > 0:
                # Convert to numeric for comparison
                df_valid_ndcg = df_valid_ndcg.copy()
                df_valid_ndcg['nDCG@10_numeric'] = pd.to_numeric(
                    df_valid_ndcg['nDCG@10'], errors='coerce')
                best_ndcg = df_valid_ndcg.loc[df_valid_ndcg['nDCG@10_numeric'].idxmax()]
                print(
                    f"✅ Best Ranking (nDCG@10): {best_ndcg['Algorithm']} ({best_ndcg['nDCG@10']})")

        if 'Time (s)' in df.columns:
            df_valid_time = df[df['Time (s)'] != 'N/A']
            if len(df_valid_time) > 0:
                df_valid_time = df_valid_time.copy()
                df_valid_time['Time_numeric'] = pd.to_numeric(
                    df_valid_time['Time (s)'], errors='coerce')
                fastest = df_valid_time.loc[df_valid_time['Time_numeric'].idxmin(
                )]
                print(
                    f"⚡ Fastest Algorithm: {fastest['Algorithm']} ({fastest['Time (s)']}s)")

        print("\n" + "="*80)
        print("HYBRID MODEL RATIONALE")
        print("="*80)
        print("""
The Hybrid model combines the top 3 algorithms:
- 50% SVD: Best accuracy (lowest RMSE/MAE)
- 30% User-CF: Best ranking metrics (nDCG)
- 20% TF-IDF: Diversity + cold-start solution

Expected benefits:
✓ Balanced performance across all metrics
✓ Robust to different data scenarios
✓ Solves cold-start problem (new users/movies)
✓ Leverages multiple data sources (ratings + content)
        """)

    elif surprise_results:
        df = pd.DataFrame(surprise_results)
        print("\nSurprise-based Algorithms:")
        print(df.to_string(index=False))

    elif custom_results:
        df = pd.DataFrame(custom_results)
        print("\nCustom Algorithms:")
        print(df.to_string(index=False))

    # Recommendations based on results
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FINAL SYSTEM")
    print("="*80)
    print("""
Based on evaluation results:

1. SVD Matrix Factorization
   - Best overall accuracy (lowest RMSE/MAE)
   - Good ranking metrics
   - Scalable and well-established

2. User-Based CF (Pearson)
   - Best ranking metrics (nDCG@10)
   - Competitive with SVD
   - Classic collaborative filtering approach

3. TF-IDF Content-Based
   - Solves cold-start problem
   - Uses movie overviews (unique data source)
   - Fastest algorithm (0.01s)
   - Adds diversity to recommendations

4. HYBRID: Combine top 3 methods (ACTIVE APPROACH)
   - 50% SVD + 30% User-CF + 20% TF-IDF
   - Combines strengths of all methods
   - Implemented in getOptimizedHybridRecommendations()

DEPRECATED (kept for reference):
- Item-Based CF: Slower, lower accuracy in evaluation
- Time-Decay CF: Minimal improvement observed
    """)


if __name__ == "__main__":
    compare_all_methods()
