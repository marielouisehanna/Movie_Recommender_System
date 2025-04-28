# Movie Recommendation System

A Flask web application that recommends movies based on user ratings using different collaborative filtering algorithms.

## Overview

This application allows users to rate movies and receive personalized recommendations. It implements three different recommendation algorithms and provides performance benchmarks to compare them:

- User-User Collaborative Filtering
- Item-Item Collaborative Filtering  
- Global Baseline Estimate

## Features

- Simple movie rating interface (1-5 stars)
- Personalized movie recommendations
- Algorithm performance comparison
- Visual benchmarking charts
- Responsive design

## Project Structure

```
├── app.py                 # Flask application with routing logic
├── static/
│   └── style.css          # CSS styling
├── templates/
│   ├── home.html          # Home page with movie rating interface
│   └── benchmark.html     # Results page with recommendations and benchmarks
├── ratings.csv            # Movie ratings dataset
├── movies.csv             # Movie information dataset
└── recommender_system.py  # Recommendation algorithms implementation
```

## Setup and Installation

1. Make sure you have Python 3.7+ installed
2. Clone this repository 
3. Install the required dependencies:

```bash
pip install flask pandas numpy scipy scikit-learn matplotlib seaborn
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and go to http://127.0.0.1:5000/

## How It Works

1. **Data Loading**: The app loads movie and rating data from CSV files.
2. **User Input**: Users rate movies they've watched on a 1-5 scale.
3. **Recommendation Generation**: The system applies the selected algorithm to generate personalized recommendations.
4. **Benchmarking**: All three algorithms are evaluated on processing speed and recommendation count.
5. **Results Display**: Recommendations and performance metrics are shown to the user.

## Recommendation Algorithms

### User-User Collaborative Filtering
Finds users with similar taste to the current user and recommends movies they rated highly.

### Item-Item Collaborative Filtering
Finds movies similar to ones the user rated highly and recommends them.

### Global Baseline Estimate
Makes predictions based on global averages, user bias, and item bias.


