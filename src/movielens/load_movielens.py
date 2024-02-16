# Description: Load MovieLens data and create a dataset for training and testing.
# Code adapted from https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization

import random
import requests
import zipfile
import io
import shutil
import pandas as pd
import os
import collections
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Tuple, Optional, List

path_to_1m = "/Volumes/T7 Touch/TheseProject/FLDATA/MovieLens"
def download_movielens_data(dataset_path):
  """Downloads and copies MovieLens data to local /tmp directory."""
  if dataset_path.startswith('http'):
    r = requests.get(dataset_path)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=path_to_1m)
  else:
    os.makedirs(path_to_1m, exist_ok=True)
    for filename in ['ratings.dat', 'movies.dat', 'users.dat']:
        shutil.copy(
          os.path.join(dataset_path, filename),
          os.path.join(path_to_1m, filename),
          overwrite=True)

#download_movielens_data('http://files.grouplens.org/datasets/movielens/ml-1m.zip')

def load_movielens_data(
    data_directory: str = path_to_1m,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Loads pandas DataFrames for ratings, movies, users from data directory."""
  # Load pandas DataFrames from data directory. Assuming data is formatted as
  # specified in http://files.grouplens.org/datasets/movielens/ml-1m-README.txt.
  ratings_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "ratings.dat"),
      sep="::",
      names=["UserID", "MovieID", "Rating", "Timestamp"], engine="python")
  movies_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "movies.dat"),
      sep="::",
      names=["MovieID", "Title", "Genres"], engine="python", 
      encoding = "ISO-8859-1")

#   # Create dictionaries mapping from old IDs to new (remapped) IDs for both
#   # MovieID and UserID. Use the movies and users present in ratings_df to
#   # determine the mapping, since movies and users without ratings are unneeded.
  movie_mapping = {
      old_movie: new_movie for new_movie, old_movie in enumerate(
          ratings_df.MovieID.astype("category").cat.categories)
  }
  user_mapping = {
      old_user: new_user for new_user, old_user in enumerate(
          ratings_df.UserID.astype("category").cat.categories)
  }

  # Map each DataFrame consistently using the now-fixed mapping.
  ratings_df.MovieID = ratings_df.MovieID.map(movie_mapping)
  ratings_df.UserID = ratings_df.UserID.map(user_mapping)
  movies_df.MovieID = movies_df.MovieID.map(movie_mapping)

  # Remove nulls resulting from some movies being in movies_df but not
  # ratings_df.
  movies_df = movies_df[pd.notnull(movies_df.MovieID)]

  return ratings_df, movies_df


def plot_genre_distribution(movies_df):
    movie_genres_list = movies_df.Genres.tolist()
    # Count the number of times each genre describes a movie.
    genre_count = collections.defaultdict(int)
    for genres in movie_genres_list:
        curr_genres_list = genres.split('|')
        for genre in curr_genres_list:
            genre_count[genre] += 1
    genre_name_list, genre_count_list = zip(*genre_count.items())

    plt.figure(figsize=(5, 5))
    plt.pie(genre_count_list, labels=genre_name_list)
    plt.title('MovieLens Movie Genres')
    plt.show()
    
    
def print_top_genres_for_user(ratings_df, movies_df, user_id):
  """Prints top movie genres for user with ID user_id."""
  user_ratings_df = ratings_df[ratings_df.UserID == user_id]
  movie_ids = user_ratings_df.MovieID

  genre_count = collections.Counter()
  for movie_id in movie_ids:
    genres_string = movies_df[movies_df.MovieID == movie_id].Genres.tolist()[0]
    for genre in genres_string.split('|'):
      genre_count[genre] += 1

  print(f'\nFor user {user_id}:')
  for (genre, freq) in genre_count.most_common(5):
    print(f'{genre} was rated {freq} times')
    
    
class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.x = torch.tensor(ratings_df.MovieID.values, dtype=torch.long)
        self.y = torch.tensor(ratings_df.Rating.values, dtype=torch.float32)
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
       return self.x[index], self.y[index]
   
    def __len__(self) -> int:
        return len(self.x)
   

def create_user_datasets(ratings_df: pd.DataFrame,
                         max_examples_per_user: Optional[int] = None,
                         min_examples_per_user: int = 0,
                         max_clients: Optional[int] = None) -> List[Subset]:
    num_users = len(ratings_df.UserID.unique())
    if max_clients is not None:
        num_users = min(num_users, max_clients)
    user_datasets = []
    for i in range(num_users):
        user_ratings_df = ratings_df[ratings_df.UserID == i]
        if len(user_ratings_df) > min_examples_per_user:
            if max_examples_per_user is not None:
                n = min(len(user_ratings_df), max_examples_per_user)
                user_ratings_df = user_ratings_df.sample(n, random_state=42)
            user_data = MovieRatingDataset(user_ratings_df)
            user_datasets.append(user_data)
    return user_datasets

        
def split_dataset(user_datasets, train_frac, val_frac):
    """_summary_
    Split users to train, validation and test
    Args:
        user_datasets (_type_): List[Dataset]
        val_per (_type_): _description_
        test_per (_type_): _description_
    """
    n = len(user_datasets)
    train_idx = int(n * train_frac)
    val_idx = int(n * (train_frac + val_frac))
    train_datasets = user_datasets[:train_idx]
    val_datasets = user_datasets[train_idx:val_idx]
    test_datasets = user_datasets[val_idx:]
    return train_datasets, val_datasets, test_datasets

def create_user_dataloader(dataset,trainfrac,valfrac,batch_size):
    nb_rating_user = len(dataset)
    train_idx = int(nb_rating_user*trainfrac)
    val_idx = int(nb_rating_user*(trainfrac+valfrac))
    train_dataset = Subset(dataset, range(0,train_idx))
    val_dataset = Subset(dataset, range(train_idx,val_idx))
    test_dataset = Subset(dataset, range(val_idx,nb_rating_user))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    return train_loader, val_loader, test_loader
    


# ratings_df, movies_df = load_movielens_data(path_to_1m)
# user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
# train_users,val_users, test_users = split_dataset(user_datasets, 0.8, 0.1)

# trainloader, valloader, testloader = create_user_dataloader(user_datasets[1], 0.8, 0.1, 5)
