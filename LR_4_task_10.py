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
        return float('inf')


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


def load_ratings(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_recommendations(ratings, target_user, similarity=pearson_correlation):
    totals = {}
    sim_sums = {}

    for other in ratings:
        if other == target_user:
            continue
        sim = similarity(ratings[target_user], ratings[other])

        if sim <= 0:
            continue

        for item in ratings[other]:
            if item not in ratings[target_user] or ratings[target_user][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += ratings[other][item] * sim
                sim_sums.setdefault(item, 0)
                sim_sums[item] += sim

    rankings = [(total / sim_sums[item], item) for item, total in totals.items()]
    rankings.sort(reverse=True)
    return rankings


if __name__ == "__main__":
    ratings = load_ratings('ratings.json')

    target_user = "David Smith"

    recommendations = get_recommendations(ratings, target_user)

    print(f"Рекомендації для {target_user}:")
    for score, item in recommendations:
        print(f"{item}: {score}")
