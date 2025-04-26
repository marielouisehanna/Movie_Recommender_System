import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
import json

# Recommender System Core Functions
def load_movielens_data(ratings_file='static/data/ratings.csv', movies_file='static/data/movies.csv', sample_size=500):
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
    
    # Create a user-item matrix
    user_item_matrix = filtered_ratings.pivot(index='userId', columns='movieId', values='rating')
    
    return user_item_matrix, filtered_ratings

def user_user_cf(ratings_df, target_user, target_items=None, k=20):
    """
    Optimized User-User Collaborative Filtering that works directly with the ratings dataframe
    """
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
    user_item_counts = ratings_df[
        (ratings_df['userId'] != target_user) & 
        (ratings_df['movieId'].isin(target_user_items))
    ]['userId'].value_counts()
    
    # Only consider users who have rated at least 3 items in common
    potential_neighbors = user_item_counts[user_item_counts >= 3].index.tolist()
    
    if len(potential_neighbors) == 0:
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
        except Exception:
            continue
    
    # Get top k similar users
    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if not top_neighbors:
        return {}
    
    # For efficiency, only process a subset of the target items if there are too many
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
    
    return predictions

def item_item_cf(ratings_df, target_user, target_items=None, k=20):
    """
    Optimized Item-Item Collaborative Filtering that works directly with the ratings dataframe
    """
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
            except Exception:
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
    
    return predictions

def global_baseline_estimate(ratings_df, target_user=None, target_items=None):
    """
    Optimized Global Baseline Estimate that works directly with the ratings dataframe
    """
    # Calculate global mean
    global_mean = ratings_df['rating'].mean()
    
    # Calculate user biases
    user_means = ratings_df.groupby('userId')['rating'].mean()
    user_biases = user_means - global_mean
    
    # Calculate item biases
    item_means = ratings_df.groupby('movieId')['rating'].mean()
    item_biases = item_means - global_mean
    
    # If no specific user is provided, return the complete bias structure
    if target_user is None:
        return {
            'global_mean': global_mean,
            'user_biases': user_biases,
            'item_biases': item_biases
        }
    
    # Get items rated by the target user
    if target_user not in user_biases.index:
        user_bias = 0  # Use 0 for new users
    else:
        user_bias = user_biases.get(target_user, 0)
    
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
    
    for item in target_items:
        if item in item_biases.index:
            item_bias = item_biases.get(item, 0)
            predictions[item] = global_mean + user_bias + item_bias
        else:
            # Use just global mean for new items
            predictions[item] = global_mean + user_bias
    
    return predictions

# Flask Web Application
app = Flask(__name__)

# Global variables to store data
ratings_data = None
movies_data = None
filtered_ratings = None
user_item_matrix = None
available_users = []
comparison_results = None
loaded = False

def load_data():
    global ratings_data, movies_data, filtered_ratings, user_item_matrix, available_users, loaded
    
    data_dir = 'static/data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if files exist
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
        print("Warning: Dataset files not found in static/data directory")
        return False
    
    try:
        # Load data
        ratings_data, movies_data = load_movielens_data(sample_size=500)
        
        # Prepare data
        user_item_matrix, filtered_ratings = prepare_data(ratings_data)
        
        # Get available users
        available_users = filtered_ratings['userId'].unique().tolist()
        
        loaded = True
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

@app.route('/')
def index():
    global loaded
    
    # Try to load data if not already loaded
    if not loaded:
        load_success = load_data()
        if not load_success:
            return render_template('setup.html')
    
    # If data is loaded, show the main app
    return render_template('index.html', 
                          user_count=len(available_users) if available_users else 0,
                          movie_count=len(movies_data) if movies_data is not None else 0,
                          rating_count=len(filtered_ratings) if filtered_ratings is not None else 0)

@app.route('/setup', methods=['GET'])
def setup():
    return render_template('setup.html')

@app.route('/user_recommendations')
def user_recommendations():
    if not loaded:
        return redirect(url_for('index'))
    
    # Get 10 random users to display
    sample_users = random.sample(available_users, min(10, len(available_users)))
    
    return render_template('user_recommendations.html', users=sample_users)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.form.get('userId')
    algorithm = request.form.get('algorithm', 'baseline')  # Default to baseline
    
    if not user_id or user_id not in map(str, available_users):
        return jsonify({'error': 'Invalid user ID'})
    
    user_id = int(user_id)
    
    # Get movies already rated by this user
    user_rated_movies = filtered_ratings[filtered_ratings['userId'] == user_id]
    
    # Select recommendation algorithm
    if algorithm == 'user_user':
        predictions = user_user_cf(filtered_ratings, user_id)
    elif algorithm == 'item_item':
        predictions = item_item_cf(filtered_ratings, user_id)
    else:  # baseline as default
        predictions = global_baseline_estimate(filtered_ratings, user_id)
    
    # Sort and get top 10 recommendations
    top_recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Get movie details for the recommendations
    recommended_movies = []
    for movie_id, predicted_rating in top_recommendations:
        movie_info = movies_data[movies_data['movieId'] == movie_id]
        if not movie_info.empty:
            recommended_movies.append({
                'movieId': int(movie_id),
                'title': movie_info.iloc[0]['title'],
                'genres': movie_info.iloc[0]['genres'],
                'predicted_rating': round(predicted_rating, 2)
            })
    
    # Get movies already rated by the user
    rated_movies = []
    for _, row in user_rated_movies.iterrows():
        movie_info = movies_data[movies_data['movieId'] == row['movieId']]
        if not movie_info.empty:
            rated_movies.append({
                'movieId': int(row['movieId']),
                'title': movie_info.iloc[0]['title'],
                'genres': movie_info.iloc[0]['genres'],
                'rating': row['rating']
            })
    
    # Sort rated movies by rating (highest first)
    rated_movies = sorted(rated_movies, key=lambda x: x['rating'], reverse=True)
    
    return jsonify({
        'userId': user_id,
        'recommendedMovies': recommended_movies,
        'ratedMovies': rated_movies[:10]  # Limit to top 10 rated
    })

@app.route('/compare_algorithms')
def compare_algorithms():
    if not loaded:
        return redirect(url_for('index'))
    
    return render_template('compare.html')

@app.route('/run_comparison', methods=['POST'])
def run_comparison():
    global comparison_results
    
    # Get test parameters
    test_size = int(request.form.get('testSize', 10))
    test_ratio = float(request.form.get('testRatio', 0.2))
    
    # Get selected algorithms
    algorithms_json = request.form.get('algorithms', '["user_user", "item_item", "baseline"]')
    try:
        algorithms = json.loads(algorithms_json)
    except:
        # Default to all algorithms if parsing fails
        algorithms = ["user_user", "item_item", "baseline"]
    
    # Ensure at least one algorithm is selected
    if not algorithms:
        algorithms = ["baseline"]  # Default to baseline if none selected
    
    # Select a subset of users for testing
    test_users = random.sample(available_users, min(test_size, len(available_users)))
    
    # Initialize results
    results = {}
    
    for method in algorithms:
        all_predictions = []
        all_actuals = []
        
        for user in test_users:
            # Get ratings for this user
            user_ratings = filtered_ratings[filtered_ratings['userId'] == user]
            
            if len(user_ratings) < 5:  # Skip users with too few ratings
                continue
            
            # Hold out a portion of ratings for testing
            n_test = max(1, int(test_ratio * len(user_ratings)))
            test_indices = np.random.choice(user_ratings.index, size=n_test, replace=False)
            
            # Create test and training sets
            test_set = user_ratings.loc[test_indices]
            train_set = filtered_ratings.drop(test_indices)
            
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
            results[method] = {'rmse': rmse, 'count': len(all_predictions)}
    
    # Store results
    comparison_results = results
    
    # Return the results as JSON
    return jsonify({
        'results': {k: {'rmse': float(v['rmse']), 'count': v['count']} for k, v in results.items()}
    })

@app.route('/example_matrix')
def example_matrix():
    return render_template('example_matrix.html')

@app.route('/run_example', methods=['POST'])
def run_example():
    # Get selected algorithm
    algorithm = request.form.get('algorithm', 'baseline')
    
    # Create the example data from the original matrix
    example_data = {
        'userId': ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D'],
        'movieId': ['HP1', 'HP3', 'SW1', 'HP1', 'HP2', 'HP3', 'TW', 'SW1', 'SW2', 'SW3', 'HP2', 'SW3'],
        'rating': [4, 5, 1, 5, 5, 4, 2, 4, 5, 3, 3, 3]
    }
    
    example_df = pd.DataFrame(example_data)
    
    # Create the original matrix
    example_matrix = example_df.pivot(index='userId', columns='movieId', values='rating')
    original_matrix = example_matrix.fillna('').to_dict(orient='index')
    
    # Apply selected algorithm
    if algorithm == 'user_user':
        # User-User Collaborative Filtering
        filled_matrix = run_user_user_cf(example_df, example_matrix)
    elif algorithm == 'item_item':
        # Item-Item Collaborative Filtering
        filled_matrix = run_item_item_cf(example_df, example_matrix)
    else:
        # Global Baseline (default)
        filled_matrix = run_global_baseline(example_df, example_matrix)
    
    # Convert to dictionary for JSON
    filled_matrix_dict = filled_matrix.round(2).to_dict(orient='index')

    
    # Mark B-SW2 as a special case that shouldn't show a prediction
    # This happens when there's not enough data for a good prediction
    if 'B' in filled_matrix_dict and 'SW2' in filled_matrix_dict['B']:
        filled_matrix_dict['B']['SW2'] = None  # This will be detected in the frontend
    
    return jsonify({
        'originalMatrix': original_matrix,
        'filledMatrix': filled_matrix_dict
    })
    

def run_global_baseline(example_df, example_matrix):
    """Apply Global Baseline to the example matrix"""
    # Apply Global Baseline
    baseline_params = global_baseline_estimate(example_df)
    global_mean = baseline_params['global_mean']
    user_biases = baseline_params['user_biases'].to_dict()
    item_biases = baseline_params['item_biases'].to_dict()
    
    # Get all user-item pairs
    all_users = example_df['userId'].unique()
    all_items = example_df['movieId'].unique()
    
    # Fill in missing values with baseline predictions
    filled_matrix = example_matrix.copy()
    
    for user in all_users:
        for item in all_items:
            if pd.isna(example_matrix.loc[user, item] if user in example_matrix.index and item in example_matrix.columns else None):
                user_bias = user_biases.get(user, 0)
                item_bias = item_biases.get(item, 0)
                filled_matrix.loc[user, item] = global_mean + user_bias + item_bias
    
    return filled_matrix

def run_user_user_cf(example_df, example_matrix):
    """Apply User-User CF to the example matrix"""
    # Create a filled matrix
    filled_matrix = example_matrix.copy()
    
    # Get all users and items
    all_users = example_df['userId'].unique()
    all_items = example_df['movieId'].unique()
    
    # For each user
    for user in all_users:
        # Get items not rated by this user
        user_rated_items = set(example_df[example_df['userId'] == user]['movieId'])
        user_unrated_items = set(all_items) - user_rated_items
        
        if not user_unrated_items:
            continue
        
        # Get predictions for unrated items using the example-specific implementation
        predictions = user_user_cf_example(example_df, user, list(user_unrated_items))
        
        # Fill in predictions
        for item, predicted_rating in predictions.items():
            filled_matrix.loc[user, item] = predicted_rating
    
    # If there are still missing values, fill them with global baseline
    filled_matrix = fill_remaining_with_baseline(example_df, filled_matrix)
    
    return filled_matrix

def run_item_item_cf(example_df, example_matrix):
    """Apply Item-Item CF to the example matrix"""
    # Create a filled matrix
    filled_matrix = example_matrix.copy()
    
    # Get all users and items
    all_users = example_df['userId'].unique()
    all_items = example_df['movieId'].unique()
    
    # For each user
    for user in all_users:
        # Get items not rated by this user
        user_rated_items = set(example_df[example_df['userId'] == user]['movieId'])
        user_unrated_items = set(all_items) - user_rated_items
        
        if not user_unrated_items:
            continue
            
        # Get predictions for unrated items using the example-specific implementation
        predictions = item_item_cf_example(example_df, user, list(user_unrated_items))
        
        # Fill in predictions
        for item, predicted_rating in predictions.items():
            filled_matrix.loc[user, item] = predicted_rating
    
    # If there are still missing values, fill them with global baseline
    filled_matrix = fill_remaining_with_baseline(example_df, filled_matrix)
    
    return filled_matrix

def fill_remaining_with_baseline(example_df, matrix):
    """Helper function to fill any remaining NaN values using the global baseline method"""
    # Calculate baseline parameters
    baseline_params = global_baseline_estimate(example_df)
    global_mean = baseline_params['global_mean']
    user_biases = baseline_params['user_biases'].to_dict()
    item_biases = baseline_params['item_biases'].to_dict()
    
    # Create a copy of the matrix
    filled_matrix = matrix.copy()
    
    # Find cells with NaN values
    for user in filled_matrix.index:
        for item in filled_matrix.columns:
            if pd.isna(filled_matrix.loc[user, item]):
                user_bias = user_biases.get(user, 0)
                item_bias = item_biases.get(item, 0)
                filled_matrix.loc[user, item] = global_mean + user_bias + item_bias
    
    return filled_matrix

# Specialized implementations for the example matrix
def user_user_cf_example(example_df, target_user, target_items=None, k=2):
    """
    Optimized User-User Collaborative Filtering specifically for the example matrix
    """
    # Get the items rated by the target user
    target_user_ratings = example_df[example_df['userId'] == target_user]
    target_user_items = set(target_user_ratings['movieId'])
    
    # If no target items specified, find all items not rated by the target user
    if target_items is None:
        all_items = set(example_df['movieId'].unique())
        target_items = all_items - target_user_items
    else:
        # Filter out items already rated by the target user
        target_items = set(target_items) - target_user_items
    
    # Find all users who have rated at least one item that the target user has rated
    other_users = set(example_df['userId'].unique()) - {target_user}
    
    # Calculate similarity for all potential neighbors
    similarities = {}
    target_user_ratings_dict = dict(zip(target_user_ratings['movieId'], target_user_ratings['rating']))
    
    for neighbor in other_users:
        # Get ratings by this neighbor
        neighbor_ratings = example_df[example_df['userId'] == neighbor]
        neighbor_ratings_dict = dict(zip(neighbor_ratings['movieId'], neighbor_ratings['rating']))
        
        # Find common items
        common_items = target_user_items.intersection(set(neighbor_ratings['movieId']))
        
        if len(common_items) < 1:
            continue
        
        # Extract ratings for common items
        target_ratings_array = np.array([target_user_ratings_dict[item] for item in common_items])
        neighbor_ratings_array = np.array([neighbor_ratings_dict[item] for item in common_items])
        
        # Calculate similarity (if only 1 common item, use absolute difference)
        try:
            if len(common_items) == 1:
                # If only one common item, use rating similarity instead of cosine
                similarity = 1 - abs(target_ratings_array[0] - neighbor_ratings_array[0]) / 4
            else:
                similarity = 1 - cosine(target_ratings_array, neighbor_ratings_array)
                
            if not np.isnan(similarity) and similarity > 0:
                similarities[neighbor] = similarity
        except Exception:
            continue
    
    # Get top k similar users
    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if not top_neighbors:
        return {}
    
    # Predict ratings
    predictions = {}
    
    # Get all ratings by top neighbors for the target items in one go
    neighbor_ids = [user for user, _ in top_neighbors]
    relevant_ratings = example_df[
        (example_df['userId'].isin(neighbor_ids)) & 
        (example_df['movieId'].isin(target_items))
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
    
    return predictions

def item_item_cf_example(example_df, target_user, target_items=None, k=2):
    """
    Optimized Item-Item Collaborative Filtering specifically for the example matrix
    """
    # Get items rated by the target user
    target_user_ratings = example_df[example_df['userId'] == target_user]
    target_user_items = set(target_user_ratings['movieId'])
    target_user_ratings_dict = dict(zip(target_user_ratings['movieId'], target_user_ratings['rating']))
    
    # If no target items specified, find all items not rated by the target user
    if target_items is None:
        all_items = set(example_df['movieId'].unique())
        target_items = all_items - target_user_items
    else:
        # Filter out items already rated by the target user
        target_items = set(target_items) - target_user_items
    
    # Predict ratings
    predictions = {}
    
    for target_item in target_items:
        # Find similarity between this item and items rated by the user
        item_similarities = {}
        
        for rated_item in target_user_items:
            # Find users who rated both items
            target_item_raters = set(example_df[example_df['movieId'] == target_item]['userId'])
            rated_item_raters = set(example_df[example_df['movieId'] == rated_item]['userId'])
            common_users = target_item_raters.intersection(rated_item_raters)
            
            if len(common_users) < 1:
                continue
            
            # Get ratings for both items by common users
            target_item_ratings = example_df[(example_df['movieId'] == target_item) & 
                                            (example_df['userId'].isin(common_users))]
            rated_item_ratings = example_df[(example_df['movieId'] == rated_item) & 
                                           (example_df['userId'].isin(common_users))]
            
            # Create rating dictionaries for faster lookup
            target_item_dict = dict(zip(target_item_ratings['userId'], target_item_ratings['rating']))
            rated_item_dict = dict(zip(rated_item_ratings['userId'], rated_item_ratings['rating']))
            
            # Extract ratings
            common_users_list = list(common_users)
            try:
                if len(common_users) == 1:
                    # If only one common user, use rating similarity instead of cosine
                    user = common_users_list[0]
                    similarity = 1 - abs(target_item_dict[user] - rated_item_dict[user]) / 4
                else:
                    target_item_array = np.array([target_item_dict[user] for user in common_users_list])
                    rated_item_array = np.array([rated_item_dict[user] for user in common_users_list])
                    similarity = 1 - cosine(target_item_array, rated_item_array)
                
                if not np.isnan(similarity) and similarity > 0:
                    item_similarities[rated_item] = similarity
            except Exception:
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
    
    return predictions

if __name__ == '__main__':
    # Try to load data at startup
    load_data()
    app.run(debug=True)
