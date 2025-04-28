from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from urllib.parse import quote

# Import functions from recommender_system.py
from recommender_system import (
    load_movielens_data, 
    prepare_data, 
    user_user_cf, 
    item_item_cf,
    global_baseline_estimate,
    evaluate_recommendations
)

app = Flask(__name__)

# Load and prepare data once at startup
print("Loading data...")
ratings, movies = load_movielens_data(sample_size=200)  # Use a smaller sample for demonstration
_, filtered_ratings = prepare_data(ratings, min_ratings_per_user=3, min_ratings_per_item=3)

# Create a mapping from movie ID to movie title for display purposes
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

# Get a list of popular/well-rated movies to show on the recommendation page
def get_popular_movies(count=20):
    # Calculate average rating and count of ratings for each movie
    movie_stats = filtered_ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        count=('rating', 'count')
    ).reset_index()
    
    # Filter movies with at least 5 ratings
    movie_stats = movie_stats[movie_stats['count'] >= 5]
    
    # Sort by average rating (descending) and then by count (descending)
    popular_movies = movie_stats.sort_values(['avg_rating', 'count'], ascending=[False, False]).head(count)
    
    # Join with movie titles
    popular_movies = popular_movies.merge(movies, on='movieId')
    
    return popular_movies[['movieId', 'title', 'avg_rating']].to_dict('records')

# Generate recommendations for a user based on their ratings
def generate_recommendations(user_ratings, method='user_user'):
    """
    Generate movie recommendations based on user ratings
    
    Parameters:
    - user_ratings: Dictionary mapping movieId to rating
    - method: Recommendation method ('user_user', 'item_item', or 'baseline')
    
    Returns:
    - List of recommended movies with scores
    """
    start_time = time.time()
    
    # Create a temporary user ID for the current session
    temp_user_id = 999999
    
    # Create a temporary dataframe with the user's ratings
    temp_ratings = pd.DataFrame({
        'userId': [temp_user_id] * len(user_ratings),
        'movieId': list(user_ratings.keys()),
        'rating': list(user_ratings.values())
    })
    
    # Combine with the existing ratings data
    combined_ratings = pd.concat([filtered_ratings, temp_ratings])
    
    # Get predictions based on the selected method
    if method == 'user_user':
        predictions = user_user_cf(combined_ratings, temp_user_id, k=10)
    elif method == 'item_item':
        predictions = item_item_cf(combined_ratings, temp_user_id, k=10)
    else:  # baseline
        predictions = global_baseline_estimate(combined_ratings, temp_user_id)
    
    # Convert predictions to a list of (movieId, predicted_rating) tuples
    recommendations = [(int(movie_id), score) for movie_id, score in predictions.items()]
    
    # Sort by predicted rating (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 10 recommendations
    recommendations = recommendations[:10]
    
    # Add movie titles
    recommendations_with_titles = []
    for movie_id, score in recommendations:
        title = movie_id_to_title.get(movie_id, f"Movie {movie_id}")
        recommendations_with_titles.append({
            'id': movie_id,
            'title': title,
            'score': round(score, 2)
        })
    
    processing_time = time.time() - start_time
    
    return recommendations_with_titles, processing_time

# Create a benchmark comparison chart
def create_benchmark_chart(user_ratings):
    """Generate a comparison chart for different recommendation methods"""
    
    # Use the same approach as generate_recommendations but for all methods
    temp_user_id = 999999
    temp_ratings = pd.DataFrame({
        'userId': [temp_user_id] * len(user_ratings),
        'movieId': list(user_ratings.keys()),
        'rating': list(user_ratings.values())
    })
    combined_ratings = pd.concat([filtered_ratings, temp_ratings])
    
    # Evaluate all methods and measure time
    methods = ['user_user', 'item_item', 'baseline']
    performance = {}
    
    for method in methods:
        start_time = time.time()
        if method == 'user_user':
            predictions = user_user_cf(combined_ratings, temp_user_id, k=10)
        elif method == 'item_item':
            predictions = item_item_cf(combined_ratings, temp_user_id, k=10)
        else:  # baseline
            predictions = global_baseline_estimate(combined_ratings, temp_user_id)
        
        processing_time = time.time() - start_time
        
        # For the demo, use processing time as a metric
        # In a real system, you would use RMSE from evaluate_recommendations
        performance[method] = {
            'processing_time': round(processing_time, 3),
            'num_recommendations': len(predictions)
        }
    
    # Create a bar chart for processing time with improved styling
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-pastel')
    
    # Define better colors
    colors = ['#3a5bd8', '#4e69e0', '#6077e3']
    
    # Create the first subplot for processing times
    plt.subplot(1, 2, 1)
    method_labels = ['User-User CF', 'Item-Item CF', 'Global Baseline']
    times = [performance[m]['processing_time'] for m in methods]
    
    bars = plt.bar(method_labels, times, color=colors, width=0.6)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.title('Processing Time Comparison', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(times) * 1.2)  # Add some headroom for labels
    
    # Create the second subplot for number of recommendations
    plt.subplot(1, 2, 2)
    counts = [performance[m]['num_recommendations'] for m in methods]
    
    bars = plt.bar(method_labels, counts, color=colors, width=0.6)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Number of Recommendations', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(counts) * 1.2)  # Add some headroom for labels
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.15)
    
    # Convert plot to base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    encoded = base64.b64encode(image_png).decode('utf-8')
    chart_data = f"data:image/png;base64,{encoded}"
    
    return chart_data, performance

# Routes
@app.route('/')
def home():
    popular_movies = get_popular_movies(20)
    return render_template('home.html', movies=popular_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user ratings from form
    user_ratings = {}
    for key, value in request.form.items():
        if key.startswith('rating_') and value:
            movie_id = int(key.split('_')[1])
            user_ratings[movie_id] = float(value)
    
    if not user_ratings:
        return redirect(url_for('home'))
    
    # Get recommendation method
    method = request.form.get('method', 'user_user')
    
    # Generate recommendations
    recommendations, processing_time = generate_recommendations(user_ratings, method)
    
    # Create benchmark chart
    chart_data, performance = create_benchmark_chart(user_ratings)
    
    # Get the titles of the rated movies
    rated_movies = []
    for movie_id, rating in user_ratings.items():
        title = movie_id_to_title.get(movie_id, f"Movie {movie_id}")
        rated_movies.append({
            'id': movie_id,
            'title': title,
            'rating': rating
        })
    
    return render_template(
        'benchmark.html',
        recommendations=recommendations,
        rated_movies=rated_movies,
        method=method,
        processing_time=round(processing_time, 3),
        chart_data=chart_data,
        performance=performance
    )

if __name__ == '__main__':
    app.run(debug=True)