import json
import numpy as np
from collections import defaultdict

with open('ratings.json', 'r') as file:
    data = json.load(file)


def pearson_correlation(user1, user2, ratings):
    common_movies = set(ratings[user1].keys()).intersection(set(ratings[user2].keys()))
    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    user1_ratings = np.array([ratings[user1][movie] for movie in common_movies])
    user2_ratings = np.array([ratings[user2][movie] for movie in common_movies])

    mean1 = np.mean(user1_ratings)
    mean2 = np.mean(user2_ratings)

    numerator = np.sum((user1_ratings - mean1) * (user2_ratings - mean2))
    denominator = np.sqrt(np.sum((user1_ratings - mean1) ** 2)) * np.sqrt(np.sum((user2_ratings - mean2) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator


def get_recommendations(target_user, ratings, num_recommendations=3):
    similarity_scores = []

    for user in ratings:
        if user != target_user:
            score = pearson_correlation(target_user, user, ratings)
            similarity_scores.append((user, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Similarity scores for {target_user}: {similarity_scores}")

    top_users = [user for user, score in similarity_scores[:num_recommendations]]

    movie_scores = defaultdict(float)
    similarity_sum = defaultdict(float)

    for user in top_users:
        similarity = pearson_correlation(target_user, user, ratings)

        for movie, rating in ratings[user].items():
            if movie not in ratings[target_user]:
                movie_scores[movie] += similarity * rating
                similarity_sum[movie] += similarity

    recommendations = [(movie, score / similarity_sum[movie]) for movie, score in movie_scores.items()]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations


recommendations = get_recommendations('Bill Duffy', data)
print(f"Recommendations for Bill Duffy: {recommendations}")

recommendations = get_recommendations('Clarissa Jackson', data)
print(f"Recommendations for Clarissa Jackson: {recommendations}")
