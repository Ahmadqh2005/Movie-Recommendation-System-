import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask("movie-recommender")
app.json.compact = False

# ---------------------------------------------------------
# 1. Load Artifacts into RAM Once at Startup
# ---------------------------------------------------------
print("Loading model artifacts into memory...")
catalog = pd.read_parquet("catalog.parquet")
embeddings = np.load("embeddings.npy")

# Build a Python dictionary mapping lowercase title -> integer row index.
# This prevents pandas Series duplicates when a title exists in multiple years (e.g. Batman).
title_to_idx = {}
for row_idx, title in catalog["title"].items():
    clean = str(title).strip().lower()
    if clean not in title_to_idx:
        title_to_idx[clean] = int(row_idx)

print(f"Artifacts loaded. Catalog contains {len(catalog):,} movies.")


# ---------------------------------------------------------
# 2. Recommender Core Function
# ---------------------------------------------------------
def get_recommendations(movie_title: str, top_k: int = 5):
    clean_title = movie_title.strip().lower()

    if clean_title not in title_to_idx:
        return None

    # Guaranteed single integer index
    idx = title_to_idx[clean_title]
    target_vector = embeddings[idx]

    # Vector dot-product (cosine similarity)
    similarity_scores = np.dot(embeddings, target_vector)

    # Rank highest scores, excluding the query movie itself
    ranked_indices = np.argsort(similarity_scores)[-(top_k + 1) : -1][::-1]

    # Safely select available metadata columns
    desired_cols = ["title", "year", "vote_average", "vote_count"]
    cols_to_use = [col for col in desired_cols if col in catalog.columns]

    recs = catalog.iloc[ranked_indices][cols_to_use].to_dict(orient="records")

    for item, match_idx in zip(recs, ranked_indices):
        item["similarity_score"] = round(float(similarity_scores[match_idx]), 3)
        item["match_percent"] = round(float(similarity_scores[match_idx]) * 100, 1)

    return recs


# ---------------------------------------------------------
# 3. HTML / CSS Frontend Template
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation Engine</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
            display: flex;
            justify-content: center;
            padding: 40px 16px;
        }
        .container {
            width: 100%;
            max-width: 720px;
        }
        .header {
            text-align: center;
            margin-bottom: 28px;
        }
        .header h1 {
            color: #38bdf8;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .header p {
            color: #94a3b8;
            font-size: 14px;
        }
        .search-card {
            background-color: #1e293b;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            margin-bottom: 24px;
            border: 1px solid #334155;
        }
        form {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        input[type="text"] {
            flex: 1;
            min-width: 240px;
            padding: 12px 16px;
            font-size: 15px;
            border-radius: 8px;
            border: 1px solid #475569;
            background: #0f172a;
            color: #f8fafc;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #38bdf8;
        }
        select {
            padding: 12px 14px;
            font-size: 15px;
            border-radius: 8px;
            border: 1px solid #475569;
            background: #0f172a;
            color: #f8fafc;
        }
        button {
            padding: 12px 24px;
            font-size: 15px;
            font-weight: 600;
            background-color: #38bdf8;
            color: #0f172a;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.15s ease;
        }
        button:hover {
            background-color: #0ea5e9;
        }
        .results-box {
            background-color: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 20px;
        }
        .results-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: #cbd5e1;
        }
        .movie-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            background-color: #0f172a;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #38bdf8;
        }
        .movie-card:last-child {
            margin-bottom: 0;
        }
        .movie-info h4 {
            font-size: 16px;
            color: #f1f5f9;
            margin-bottom: 4px;
        }
        .movie-meta {
            font-size: 13px;
            color: #94a3b8;
        }
        .match-badge {
            text-align: right;
        }
        .badge-value {
            font-size: 18px;
            font-weight: 700;
            color: #4ade80;
        }
        .badge-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
        }
        .error-card {
            background-color: #450a0a;
            color: #fca5a5;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #7f1d1d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Movie Recommender</h1>
            <p>Vector Embedding & Cosine Similarity Service</p>
        </div>

        <div class="search-card">
            <form method="GET" action="/">
                <input 
                    type="text" 
                    name="title" 
                    placeholder="Enter movie title (e.g. Batman, Toy Story, Inception)..." 
                    value="{{ query or '' }}" 
                    required
                >
                <select name="top_k">
                    <option value="5" {% if top_k == 5 %}selected{% endif %}>Top 5</option>
                    <option value="10" {% if top_k == 10 %}selected{% endif %}>Top 10</option>
                </select>
                <button type="submit">Recommend</button>
            </form>
        </div>

        {% if error %}
            <div class="error-card">{{ error }}</div>
        {% endif %}

        {% if results %}
            <div class="results-box">
                <div class="results-title">Recommendations for &ldquo;{{ query }}&rdquo;</div>
                {% for movie in results %}
                <div class="movie-card">
                    <div class="movie-info">
                        <h4>{{ movie.title }}</h4>
                        <div class="movie-meta">
                            {% if movie.year %}Year: {{ movie.year }} &bull;{% endif %}
                            {% if movie.vote_average %}Rating: &#9733; {{ movie.vote_average }}{% endif %}
                            {% if movie.vote_count %}({{ movie.vote_count }} votes){% endif %}
                        </div>
                    </div>
                    <div class="match-badge">
                        <div class="badge-value">{{ movie.match_percent }}%</div>
                        <div class="badge-label">Similarity</div>
                    </div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


# ---------------------------------------------------------
# 4. Web UI Endpoint
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def ui():
    query_title = request.args.get("title")
    top_k = request.args.get("top_k", default=5, type=int)

    if not query_title:
        return render_template_string(
            HTML_TEMPLATE, query=None, results=None, error=None, top_k=top_k
        )

    results = get_recommendations(query_title, top_k=top_k)

    if results is None:
        return render_template_string(
            HTML_TEMPLATE,
            query=query_title,
            results=None,
            error=f"Movie '{query_title}' was not found in catalog.",
            top_k=top_k,
        )

    return render_template_string(
        HTML_TEMPLATE,
        query=query_title,
        results=results,
        error=None,
        top_k=top_k,
    )


# ---------------------------------------------------------
# 5. Programmatic JSON API Endpoints
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "total_movies": len(catalog)}), 200


@app.route("/recommend", methods=["GET"])
def recommend_api():
    query_title = request.args.get("title")
    top_k = request.args.get("top_k", default=5, type=int)

    if not query_title:
        return jsonify({"error": "Missing required query parameter: 'title'"}), 400

    results = get_recommendations(query_title, top_k=top_k)

    if results is None:
        return (
            jsonify({"error": f"Movie '{query_title}' not found in catalog."}),
            404,
        )

    return (
        jsonify(
            {
                "query": query_title,
                "count": len(results),
                "recommendations": results,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)