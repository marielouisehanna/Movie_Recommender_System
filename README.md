# MovieLens Recommender System

## 🎬 MovieLens Recommender System
A web-based movie recommendation system powered by Flask and built on the popular MovieLens dataset. Explore personalized movie recommendations, compare recommendation algorithms, and understand how collaborative filtering works — all from your browser.

## 🚀 Features
- 🎯 User Recommendations: Get personalized suggestions using different collaborative filtering techniques.

- ⚖️ Algorithm Comparison: Compare accuracy and performance across multiple recommendation algorithms.

- 🔍 Example Matrix: Visualize how the system fills in missing ratings using algorithmic predictions.

## Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

---

## Project Structure

```
movie_recommender/
├── app.py                   # Main Flask application
├── static/
│   ├── css/
│   │   └── style.css        # CSS styling
│   ├── js/
│   │   └── main.js          # JavaScript functionality
│   ├── img/
│   │   └── movie-illustration.svg  # SVG illustration
│   └── data/                # Directory for dataset files
├── templates/
│   ├── index.html           # Home page
│   ├── user_recommendations.html  # User recommendations page
│   ├── compare.html         # Algorithm comparison page
│   ├── example_matrix.html  # Example matrix page
│   └── setup.html           # Setup instructions page
```

---

###  Run the Application
```bash
# Make sure you're in the project directory
python app.py
```
Visit `http://127.0.0.1:5000/` in your browser to use the app.

---

## Web Application Features
The web interface offers the following capabilities:

- **User Recommendations**: Personalized movie suggestions using various algorithms.
- **Algorithm Comparison**: Analyze and compare performance of different recommendation methods.
- **Example Matrix**: Visual example of how algorithms complete a ratings matrix.
- **Setup Instructions**: Guidance for resolving missing dataset files.

---

## Recommendation Algorithms
This system includes the following algorithms:

1. **User-User Collaborative Filtering**  
   Suggests movies based on preferences of similar users.

2. **Item-Item Collaborative Filtering**  
   Recommends items similar to those already rated highly.

3. **Global Baseline Estimate**  
   Predicts ratings using global averages and user/item bias.

---

## Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Lower values indicate better accuracy.
- **Prediction Time**: Global Baseline is fastest; Item-Item CF is most computationally intensive.
- **Rating Scale**: 1 to 5 stars.

---
