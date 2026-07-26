import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

print("Loading the MovieLens 100k dataset (will download if not present)...")
# Load the movielens-100k dataset (download it if needed)
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

print("Training SVD (Singular Value Decomposition) algorithm...")
# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

print("Evaluating on test set...")
# Predict ratings for the testset
predictions = algo.test(testset)

# Then compute RMSE
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

# Predict rating for a specific user and item
uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

print(f"\nPredicting rating for user {uid} on movie {iid}...")
# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
print(f"Predicted rating: {pred.est:.2f}")

# Get top 5 recommendations for user 196
print("\nGenerating top 5 movie recommendations for user 196...")
from collections import defaultdict

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=5)

for iid, rating in top_n[uid]:
    print(f"Movie ID: {iid}, Predicted Rating: {rating:.2f}")

print("\nRecommendation system successfully trained and evaluated.")
