import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Part 1: Load and prepare the MovieLens dataset
def load_movielens_data(ratings_file='ratings.csv', movies_file='movies.csv', sample_size=None):
    """
    Load the MovieLens dataset with optional sampling for faster processing
    """
    print("Loading ratings data...")
    # Load ratings with optional sampling
    ratings = pd.read_csv(ratings_file)
    
    if sample_size and sample_size < len(ratings['userId'].unique()):
        print(f"Sampling {sample_size} users from {len(ratings['userId'].unique())} total users")
        user_ids = ratings['userId'].unique()
        sampled_users = np.random.choice(user_ids, size=sample_size, replace=False)
        ratings = ratings[ratings['userId'].isin(sampled_users)]
        print(f"Sampled dataset has {len(ratings)} ratings")
    
    # Load movies data
    print("Loading movies data...")
    movies = pd.read_csv(movies_file)
    
    return ratings, movies

def prepare_data(ratings, min_ratings_per_user=5, min_ratings_per_item=5):
    """
    Prepare the rating data for the recommender system
    Filter users and items with too few ratings
    """
    print("Preparing data...")
    
    # Filter users and items with minimum number of ratings
    user_counts = ratings['userId'].value_counts()
    item_counts = ratings['movieId'].value_counts()
    
    # Keep only users and items with minimum number of ratings
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    valid_items = item_counts[item_counts >= min_ratings_per_item].index
    
    # Filter the ratings dataframe
    filtered_ratings = ratings[
        ratings['userId'].isin(valid_users) & 
        ratings['movieId'].isin(valid_items)
    ]
    
    print(f"After filtering: {len(filtered_ratings)} ratings from {len(valid_users)} users for {len(valid_items)} items")
    
    # Create a user-item matrix (this can be memory-intensive for large datasets)
    print("Creating user-item matrix...")
    user_item_matrix = filtered_ratings.pivot(index='userId', columns='movieId', values='rating')
    
    return user_item_matrix, filtered_ratings

# Optimized version of User-User Collaborative Filtering
def user_user_cf(ratings_df, target_user, target_items=None, k=20):
    """
    Optimized User-User Collaborative Filtering that works directly with the ratings dataframe
    instead of a potentially huge user-item matrix
    
    Parameters:
    - ratings_df: DataFrame with userId, movieId, rating columns
    - target_user: The user for whom we want to predict ratings
    - target_items: List of items to predict (if None, predict all unrated items)
    - k: Number of similar users to consider
    
    Returns:
    - Dictionary with predicted ratings for the target user
    """
    start_time = time.time()
    print(f"Starting user-user CF for user {target_user}...")
    
    # Get the items rated by the target user
    target_user_ratings = ratings_df[ratings_df['userId'] == target_user]
    target_user_items = set(target_user_ratings['movieId'])
    
    # If no target items specified, find all items not rated by the target user
    if target_items is None:
        all_items = set(ratings_df['movieId'].unique())
        target_items = all_items - target_user_items
    else:
        # Filter out items already rated by the target user
        target_items = set(target_items) - target_user_items
    
    # Find all users who have rated at least one item that the target user has rated
    # This is more efficient than comparing with all users
    user_item_counts = ratings_df[
        (ratings_df['userId'] != target_user) & 
        (ratings_df['movieId'].isin(target_user_items))
    ]['userId'].value_counts()
    
    # Only consider users who have rated at least 3 items in common
    potential_neighbors = user_item_counts[user_item_counts >= 3].index.tolist()
    
    if len(potential_neighbors) == 0:
        print("No suitable neighbors found with sufficient overlap.")
        return {}
    
    # Calculate similarity for potential neighbors
    similarities = {}
    target_user_ratings_dict = dict(zip(target_user_ratings['movieId'], target_user_ratings['rating']))
    
    for neighbor in potential_neighbors:
        # Get ratings by this neighbor
        neighbor_ratings = ratings_df[ratings_df['userId'] == neighbor]
        neighbor_ratings_dict = dict(zip(neighbor_ratings['movieId'], neighbor_ratings['rating']))
        
        # Find common items
        common_items = target_user_items.intersection(set(neighbor_ratings['movieId']))
        
        if len(common_items) < 3:
            continue
        
        # Extract ratings for common items
        target_ratings_array = np.array([target_user_ratings_dict[item] for item in common_items])
        neighbor_ratings_array = np.array([neighbor_ratings_dict[item] for item in common_items])
        
        # Calculate similarity
        try:
            similarity = 1 - cosine(target_ratings_array, neighbor_ratings_array)
            if not np.isnan(similarity) and similarity > 0:
                similarities[neighbor] = similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            continue
    
    # Get top k similar users
    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if not top_neighbors:
        print("No positively correlated neighbors found.")
        return {}
    
    # For efficiency, only process a subset of the unrated items if there are too many
    if len(target_items) > 100:
        # Get the most popular items from the target items
        item_popularity = ratings_df[ratings_df['movieId'].isin(target_items)]['movieId'].value_counts()
        # Take top 100 most popular items
        target_items = set(item_popularity.head(100).index)
    
    # Predict ratings
    predictions = {}
    
    # Get all ratings by top neighbors for the target items in one go
    neighbor_ids = [user for user, _ in top_neighbors]
    relevant_ratings = ratings_df[
        (ratings_df['userId'].isin(neighbor_ids)) & 
        (ratings_df['movieId'].isin(target_items))
    ]
    
    # Group by movieId to predict each item
    for item in target_items:
        item_ratings = relevant_ratings[relevant_ratings['movieId'] == item]
        if len(item_ratings) == 0:
            continue
        
        weighted_sum = 0
        similarity_sum = 0
        
        for neighbor_id, similarity in top_neighbors:
            # Find the rating from this neighbor for this item
            neighbor_rating = item_ratings[item_ratings['userId'] == neighbor_id]
            if not neighbor_rating.empty:
                weighted_sum += similarity * neighbor_rating.iloc[0]['rating']
                similarity_sum += similarity
        
        if similarity_sum > 0:
            predictions[item] = weighted_sum / similarity_sum
    
    print(f"User-user CF completed in {time.time() - start_time:.2f} seconds. Predicted {len(predictions)} items.")
    return predictions

# Optimized version of Item-Item Collaborative Filtering
def item_item_cf(ratings_df, target_user, target_items=None, k=20):
    """
    Optimized Item-Item Collaborative Filtering that works directly with the ratings dataframe
    
    Parameters:
    - ratings_df: DataFrame with userId, movieId, rating columns
    - target_user: The user for whom we want to predict ratings
    - target_items: List of items to predict (if None, predict all unrated items)
    - k: Number of similar items to consider
    
    Returns:
    - Dictionary with predicted ratings for the target user
    """
    start_time = time.time()
    print(f"Starting item-item CF for user {target_user}...")
    
    # Get items rated by the target user
    target_user_ratings = ratings_df[ratings_df['userId'] == target_user]
    target_user_items = set(target_user_ratings['movieId'])
    target_user_ratings_dict = dict(zip(target_user_ratings['movieId'], target_user_ratings['rating']))
    
    # If no target items specified, find all items not rated by the target user
    if target_items is None:
        all_items = set(ratings_df['movieId'].unique())
        target_items = all_items - target_user_items
    else:
        # Filter out items already rated by the target user
        target_items = set(target_items) - target_user_items
    
    # For efficiency, limit the number of target items if there are too many
    if len(target_items) > 100:
        item_popularity = ratings_df[ratings_df['movieId'].isin(target_items)]['movieId'].value_counts()
        target_items = set(item_popularity.head(100).index)
    
    # Predict ratings
    predictions = {}
    
    for target_item in target_items:
        # Get users who rated this item
        item_raters = set(ratings_df[ratings_df['movieId'] == target_item]['userId'])
        
        # Skip if no one has rated this item
        if len(item_raters) == 0:
            continue
        
        # Find similarities between this item and items rated by the target user
        item_similarities = {}
        
        for rated_item in target_user_items:
            # Find users who rated both items
            rated_item_raters = set(ratings_df[ratings_df['movieId'] == rated_item]['userId'])
            common_users = item_raters.intersection(rated_item_raters)
            
            if len(common_users) < 3:
                continue
            
            # Get ratings for both items by common users
            target_item_ratings = ratings_df[(ratings_df['movieId'] == target_item) & 
                                            (ratings_df['userId'].isin(common_users))]
            rated_item_ratings = ratings_df[(ratings_df['movieId'] == rated_item) & 
                                           (ratings_df['userId'].isin(common_users))]
            
            # Create rating dictionaries for faster lookup
            target_item_dict = dict(zip(target_item_ratings['userId'], target_item_ratings['rating']))
            rated_item_dict = dict(zip(rated_item_ratings['userId'], rated_item_ratings['rating']))
            
            # Extract ratings
            common_users_list = list(common_users)
            try:
                target_item_array = np.array([target_item_dict[user] for user in common_users_list])
                rated_item_array = np.array([rated_item_dict[user] for user in common_users_list])
                
                # Calculate similarity
                similarity = 1 - cosine(target_item_array, rated_item_array)
                if not np.isnan(similarity) and similarity > 0:
                    item_similarities[rated_item] = similarity
            except Exception as e:
                continue
        
        # Get top k similar items
        top_items = sorted(item_similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        
        if not top_items:
            continue
        
        # Calculate prediction
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_item, similarity in top_items:
            weighted_sum += similarity * target_user_ratings_dict[similar_item]
            similarity_sum += similarity
        
        if similarity_sum > 0:
            predictions[target_item] = weighted_sum / similarity_sum
    
    print(f"Item-item CF completed in {time.time() - start_time:.2f} seconds. Predicted {len(predictions)} items.")
    return predictions

# Part 4: Implement Global Baseline Estimate
def global_baseline_estimate(ratings_df, target_user=None, target_items=None):
    """
    Optimized Global Baseline Estimate that works directly with the ratings dataframe
    
    Parameters:
    - ratings_df: DataFrame with userId, movieId, rating columns
    - target_user: The user for whom we want to predict ratings (if None, predict for all users)
    - target_items: List of items to predict (if None, predict all items)
    
    Returns:
    - Dictionary with predicted ratings for the specified user-item pairs
    """
    start_time = time.time()
    print("Computing global baseline estimates...")
    
    # Calculate global mean
    global_mean = ratings_df['rating'].mean()
    print(f"Global mean rating: {global_mean:.2f}")
    
    # Calculate user biases
    user_means = ratings_df.groupby('userId')['rating'].mean()
    user_biases = user_means - global_mean
    
    # Calculate item biases
    item_means = ratings_df.groupby('movieId')['rating'].mean()
    item_biases = item_means - global_mean
    
    # If no specific user is provided, return the complete bias structure
    if target_user is None:
        print(f"Global baseline computation completed in {time.time() - start_time:.2f} seconds.")
        return {
            'global_mean': global_mean,
            'user_biases': user_biases,
            'item_biases': item_biases
        }
    
    # Get items rated by the target user
    if target_user not in user_biases.index:
        print(f"User {target_user} not found in the dataset.")
        return {}
    
    target_user_ratings = ratings_df[ratings_df['userId'] == target_user]
    target_user_items = set(target_user_ratings['movieId'])
    
    # If no target items specified, find all items not rated by the target user
    if target_items is None:
        all_items = set(ratings_df['movieId'].unique())
        target_items = all_items - target_user_items
    else:
        # Filter out items already rated by the target user
        target_items = set(target_items) - target_user_items
    
    # Calculate predictions for specified items
    predictions = {}
    user_bias = user_biases.get(target_user, 0)
    
    for item in target_items:
        if item in item_biases.index:
            item_bias = item_biases.get(item, 0)
            predictions[item] = global_mean + user_bias + item_bias
    
    print(f"Global baseline computation completed in {time.time() - start_time:.2f} seconds. Predicted {len(predictions)} items.")
    return predictions

# Part 5: Evaluation
def evaluate_recommendations(ratings_df, test_users, test_ratio=0.2, methods=None):
    """
    Evaluate recommender systems on a set of test users
    
    Parameters:
    - ratings_df: DataFrame with userId, movieId, rating columns
    - test_users: List of users to evaluate on
    - test_ratio: Ratio of ratings to hide for testing
    - methods: List of methods to evaluate ('user_user', 'item_item', 'baseline')
    
    Returns:
    - Dictionary with RMSE scores for each method
    """
    if methods is None:
        methods = ['user_user', 'item_item', 'baseline']
    
    results = {}
    
    for method in methods:
        print(f"\nEvaluating {method} approach...")
        all_predictions = []
        all_actuals = []
        
        for user in test_users:
            print(f"Testing user {user}...")
            
            # Get ratings for this user
            user_ratings = ratings_df[ratings_df['userId'] == user]
            
            if len(user_ratings) < 5:  # Skip users with too few ratings
                print(f"User {user} has too few ratings, skipping.")
                continue
            
            # Hold out a portion of ratings for testing
            n_test = max(1, int(test_ratio * len(user_ratings)))
            test_indices = np.random.choice(user_ratings.index, size=n_test, replace=False)
            
            # Create test and training sets
            test_set = user_ratings.loc[test_indices]
            train_set = ratings_df.drop(test_indices)
            
            # Get the items to predict
            test_items = test_set['movieId'].tolist()
            
            # Get predictions based on the selected method
            predictions = {}
            
            if method == 'user_user':
                predictions = user_user_cf(train_set, user, test_items)
            elif method == 'item_item':
                predictions = item_item_cf(train_set, user, test_items)
            elif method == 'baseline':
                predictions = global_baseline_estimate(train_set, user, test_items)
            
            # Collect predictions and actual ratings
            for _, row in test_set.iterrows():
                item = row['movieId']
                if item in predictions:
                    all_predictions.append(predictions[item])
                    all_actuals.append(row['rating'])
        
        # Calculate RMSE
        if all_predictions:
            rmse = sqrt(mean_squared_error(all_actuals, all_predictions))
            results[method] = rmse
            print(f"{method} RMSE: {rmse:.4f}")
        else:
            print(f"Could not evaluate {method} approach due to insufficient predictions.")
    
    return results

def apply_to_example_matrix():
    """Apply the algorithms to the example matrix from the project specification"""
    print("\nDemonstrating on the example matrix from the project:")
    
    # Create the example data from the original matrix
    example_data = {
        'userId': ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D'],
        'movieId': ['HP1', 'HP3', 'SW1', 'HP1', 'HP2', 'HP3', 'TW', 'SW1', 'SW2', 'SW3', 'HP2', 'SW3'],
        'rating': [4, 5, 1, 5, 5, 4, 2, 4, 5, 3, 3, 3]
    }
    
    example_df = pd.DataFrame(example_data)
    
    # Print the original matrix for reference
    print("\nOriginal example matrix:")
    example_matrix = example_df.pivot(index='userId', columns='movieId', values='rating')
    print(example_matrix)
    
    # Apply all three approaches to fill in missing values
    all_users = example_df['userId'].unique()
    all_items = example_df['movieId'].unique()
    
    # Initialize result matrices
    user_user_result = example_matrix.copy()
    item_item_result = example_matrix.copy()
    baseline_result = example_matrix.copy()
    
    # Get all possible user-item pairs
    all_pairs = []
    for user in all_users:
        for item in all_items:
            if pd.isna(example_matrix.loc[user, item] if user in example_matrix.index and item in example_matrix.columns else None):
                all_pairs.append((user, item))
    
    # Apply User-User CF
    print("\nApplying User-User CF...")
    for user, _ in all_pairs:
        predictions = user_user_cf(example_df, user, k=2)  # k=2 for small dataset
        for item, rating in predictions.items():
            user_user_result.loc[user, item] = rating
    
    # Apply Item-Item CF
    print("\nApplying Item-Item CF...")
    for user, _ in all_pairs:
        predictions = item_item_cf(example_df, user, k=2)  # k=2 for small dataset
        for item, rating in predictions.items():
            item_item_result.loc[user, item] = rating
    
    # Apply Global Baseline
    print("\nApplying Global Baseline...")
    baseline_params = global_baseline_estimate(example_df)
    global_mean = baseline_params['global_mean']
    user_biases = baseline_params['user_biases']
    item_biases = baseline_params['item_biases']
    
    for user, item in all_pairs:
        user_bias = user_biases.get(user, 0)
        item_bias = item_biases.get(item, 0)
        baseline_result.loc[user, item] = global_mean + user_bias + item_bias
    
    # Display results
    print("\nMatrix filled using User-User CF:")
    print(user_user_result.round(2))
    
    print("\nMatrix filled using Item-Item CF:")
    print(item_item_result.round(2))
    
    print("\nMatrix filled using Global Baseline:")
    print(baseline_result.round(2))

def main():
    # 1. Load a small sample of MovieLens data to make computation feasible
    sample_size = 500  # Adjust based on your computational resources
    ratings, movies = load_movielens_data(sample_size=sample_size)
    
    # 2. Prepare the data
    user_item_matrix, filtered_ratings = prepare_data(ratings, min_ratings_per_user=5, min_ratings_per_item=5)
    
    # 3. Select a small set of users for testing
    active_users = user_item_matrix.index.tolist()
    test_size = min(10, len(active_users))
    test_users = np.random.choice(active_users, size=test_size, replace=False)
    print(f"Selected {test_size} users for testing")
    
    # 4. Evaluate each approach
    methods = ['user_user', 'item_item', 'baseline']
    results = evaluate_recommendations(filtered_ratings, test_users, methods=methods)
    
    # 5. Visualize results if any
    if results:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(results.keys()), y=list(results.values()))
        plt.title('Recommender System Performance Comparison')
        plt.ylabel('RMSE (lower is better)')
        plt.savefig('recommender_comparison.png')
        plt.close()
        print("Results visualization saved as 'recommender_comparison.png'")
    
    # 6. Apply to the example matrix from the project
    apply_to_example_matrix()
    
    print("\nRecommendation complete! Check the generated analysis for detailed results.")

if __name__ == "__main__":
    main()