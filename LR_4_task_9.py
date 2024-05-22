import json
import math


def euclidean_distance(user1, user2):
    common_ratings = False
    distance = 0
    for item in user1:
        if item in user2:
            distance += (user1[item] - user2[item]) ** 2
            common_ratings = True
    if common_ratings:
        return math.sqrt(distance)
    else:
        return float('inf')  # No ratings in common


def load_ratings(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    ratings = load_ratings('ratings.json')

    user_pairs = [
        ("David Smith", "Brenda Peterson"),
        ("David Smith", "Samuel Miller"),
        ("David Smith", "Julie Hammel"),
        ("David Smith", "Clarissa Jackson"),
        ("David Smith", "Adam Cohen"),
        ("David Smith", "Chris Duncan")
    ]

    for user1, user2 in user_pairs:
        distance = euclidean_distance(ratings[user1], ratings[user2])
        print(f"Евклідова відстань між {user1} та {user2}: {distance}")


def pearson_correlation(user1, user2):
    common_items = {}
    for item in user1:
        if item in user2:
            common_items[item] = 1

    n = len(common_items)
    if n == 0:
        return 0

    sum1 = sum([user1[item] for item in common_items])
    sum2 = sum([user2[item] for item in common_items])

    sum1_sq = sum([user1[item] ** 2 for item in common_items])
    sum2_sq = sum([user2[item] ** 2 for item in common_items])

    p_sum = sum([user1[item] * user2[item] for item in common_items])

    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_sq - sum1 ** 2 / n) * (sum2_sq - sum2 ** 2 / n))

    if den == 0:
        return 0

    return num / den


if __name__ == "__main__":
    for user1, user2 in user_pairs:
        correlation = pearson_correlation(ratings[user1], ratings[user2])
        print(f"Коефіцієнт кореляції Пірсона між {user1} та {user2}: {correlation}")
