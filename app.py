import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
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
    
    # Select a subset of users for testing
    test_users = random.sample(available_users, min(test_size, len(available_users)))
    
    # Initialize results
    methods = ['user_user', 'item_item', 'baseline']
    results = {}
    
    for method in methods:
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
    
    # Create visualization data
    method_labels = {
        'user_user': 'User-User CF',
        'item_item': 'Item-Item CF',
        'baseline': 'Global Baseline'
    }
    
    visualization_data = {
        'methods': [method_labels[m] for m in results.keys()],
        'rmse': [results[m]['rmse'] for m in results.keys()],
        'counts': [results[m]['count'] for m in results.keys()]
    }
    
    # Create bar chart
    fig = px.bar(
        x=visualization_data['methods'], 
        y=visualization_data['rmse'],
        title='Recommender System RMSE Comparison (Lower is Better)',
        labels={'x': 'Method', 'y': 'RMSE'},
        color=visualization_data['methods'],
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Format y-axis
    fig.update_layout(
        yaxis_range=[0, max(visualization_data['rmse']) * 1.2],
        yaxis_title_text='RMSE (Lower is Better)',
        xaxis_title_text='Method',
        title_x=0.5,
        template='plotly_white'
    )
    
    # Add value labels
    fig.update_traces(
        texttemplate='%{y:.4f}', 
        textposition='outside'
    )
    
    # Convert the figure to JSON for the frontend
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'results': {k: {'rmse': float(v['rmse']), 'count': v['count']} for k, v in results.items()},
        'chartJson': chart_json
    })

@app.route('/example_matrix')
def example_matrix():
    return render_template('example_matrix.html')

@app.route('/run_example', methods=['POST'])
def run_example():
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
    
    # Apply Global Baseline (as it works best with sparse data)
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
    
    # Convert to dictionary for JSON
    filled_matrix_dict = filled_matrix.round(2).to_dict(orient='index')
    
    # Create a heatmap for visualization
    heatmap_data = []
    
    for user in all_users:
        for item in all_items:
            if user in example_matrix.index and item in example_matrix.columns:
                if not pd.isna(example_matrix.loc[user, item]):
                    # Original value
                    heatmap_data.append({
                        'user': user,
                        'item': item,
                        'rating': float(example_matrix.loc[user, item]),
                        'type': 'Original'
                    })
                else:
                    # Predicted value
                    heatmap_data.append({
                        'user': user,
                        'item': item,
                        'rating': float(filled_matrix.loc[user, item]),
                        'type': 'Predicted'
                    })
    
    # Create heatmap
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Sort by user and item for consistent display
    user_order = ['A', 'B', 'C', 'D']
    item_order = ['HP1', 'HP2', 'HP3', 'TW', 'SW1', 'SW2', 'SW3']
    
    # Create the heatmap figure
    fig = px.density_heatmap(
        heatmap_df,
        x='item',
        y='user',
        z='rating',
        color_continuous_scale='Viridis',
        category_orders={'user': user_order, 'item': item_order},
        title='User-Item Matrix with Predicted Values',
        labels={'rating': 'Rating', 'user': 'User', 'item': 'Item'},
        text_auto=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Item',
        yaxis_title='User',
        yaxis={'categoryarray': user_order, 'categoryorder': 'array'},
        xaxis={'categoryarray': item_order, 'categoryorder': 'array'},
        title_x=0.5,
        coloraxis_colorbar=dict(title='Rating'),
        template='plotly_white'
    )
    
    # Convert the figure to JSON for the frontend
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'originalMatrix': original_matrix,
        'filledMatrix': filled_matrix_dict,
        'chartJson': chart_json
    })

if __name__ == '__main__':
    # Try to load data at startup
    load_data()
    app.run(debug=True)
