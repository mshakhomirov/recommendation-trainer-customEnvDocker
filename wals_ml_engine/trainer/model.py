
"""WALS model input data, training and predict functions."""

import datetime
import numpy as np
import os
import pandas as pd
from scipy.sparse import coo_matrix
import sh
import tensorflow as tf

import wals

import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# ratio of train set size to test set size
TEST_SET_RATIO = 10

# default hyperparameters
DEFAULT_PARAMS = {
    'weights': True,
    'latent_factors': 5,
    'num_iters': 20,
    'regularization': 0.07,
    'unobs_weight': 0.01,
    'wt_type': 0,
    'feature_wt_factor': 130.0,
    'feature_wt_exp': 0.08,
    'delimiter': '\t'
}

# parameters optimized with hypertuning
OPTIMIZED_PARAMS = {
    'latent_factors': 34,
    'regularization': 9.83,
    'unobs_weight': 0.001,
    'feature_wt_factor': 189.8,
}

# parameters optimized with hypertuning for the included web views data set
OPTIMIZED_PARAMS_WEB = {
    'latent_factors': 30,
    'regularization': 7.27,
    'unobs_weight': 0.01,
    'feature_wt_exp': 5.05,
}


def create_test_and_train_sets(args, input_file, data_type='ratings'):
  """Create test and train sets, for different input data types.

  Args:
    args: input args for job
    input_file: path to csv data file
    data_type:  
                'user_ratings': user-item ratings data

  Returns:
    array of user IDs for each row of the ratings matrix
    array of item IDs for each column of the rating matrix
    sparse coo_matrix for training
    sparse coo_matrix for test

  Raises:
    ValueError: if invalid data_type is supplied
  """
  if data_type == 'user_ratings':
    return _groups_train_and_test(input_file)
  else:
    raise ValueError('data_type arg value %s not supported.' % data_type)

def _groups_train_and_test(input_file):
  """Load page views dataset, and create train and set sparse matrices.

  Assumes 'subscriberId', 'groupId', and 'ratting' columns.

  Args:
    input_file: path to csv data file

  Returns:
    array of user IDs for each row of the ratings matrix
    array of item IDs for each column of the rating matrix
    sparse coo_matrix for training
    sparse coo_matrix for test
  """
  views_df = pd.read_csv(input_file, sep=',', header=0)

  df_items = pd.DataFrame({'groupId': views_df.groupId.unique()})
  df_sorted_items = df_items.sort_values('groupId').reset_index()
  pds_items = df_sorted_items.groupId

  # preprocess data. df.groupby.agg sorts clientId and contentId
  df_user_items = views_df.groupby(['subscriberId', 'groupId']
                                  ).agg({'rating': 'sum'})

  # create a list of (userId, itemId, timeOnPage) ratings, where userId and
  # clientId are 0-indexed
  current_u = -1
  ux = -1
  pv_ratings = []
  user_ux = []
  for timeonpg in df_user_items.itertuples():
    user = timeonpg[0][0]
    item = timeonpg[0][1]

    # as we go, build a (sorted) list of user ids
    if user != current_u:
      user_ux.append(user)
      ux += 1
      current_u = user

    # this search makes the preprocessing time O(r * i log(i)),
    # r = # ratings, i = # items
    ix = pds_items.searchsorted(item)[0]
    pv_ratings.append((ux, ix, timeonpg[1]))

  # convert ratings list and user map to np array
  pv_ratings = np.asarray(pv_ratings)
  user_ux = np.asarray(user_ux)

  # create train and test sets
  tr_sparse, test_sparse = _create_sparse_train_and_test(pv_ratings,
                                                         ux + 1,
                                                         df_items.size)

  return user_ux, pds_items.as_matrix(), tr_sparse, test_sparse

def get_user_items_w(input_file):
  """Get a JSON WOLF model where groups array mapped to subscriberId.

  """
  views_df = pd.read_csv(input_file, sep=',', header=0)
  user_items_w = views_df.groupby('subscriberId')['groupId'].apply(list).reset_index(name='items') 
  
  return user_items_w

def _create_sparse_train_and_test(ratings, n_users, n_items):
  """Given ratings, create sparse matrices for train and test sets.

  Args:
    ratings:  list of ratings tuples  (u, i, r)
    n_users:  number of users
    n_items:  number of items

  Returns:
     train, test sparse matrices in scipy coo_matrix format.
  """
  # pick a random test set of entries, sorted ascending
  test_set_size = len(ratings) / TEST_SET_RATIO
  test_set_idx = np.random.choice(xrange(len(ratings)),
                                  size=test_set_size, replace=False)
  test_set_idx = sorted(test_set_idx)

  # sift ratings into train and test sets
  ts_ratings = ratings[test_set_idx]
  tr_ratings = np.delete(ratings, test_set_idx, axis=0)

  # create training and test matrices as coo_matrix's
  u_tr, i_tr, r_tr = zip(*tr_ratings)
  tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

  u_ts, i_ts, r_ts = zip(*ts_ratings)
  test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))

  return tr_sparse, test_sparse


def train_model(args, tr_sparse):
  """Instantiate WALS model and use "simple_train" to factorize the matrix.

  Args:
    args: training args containing hyperparams
    tr_sparse: sparse training matrix

  Returns:
     the row and column factors in numpy format.
  """
  dim = args['latent_factors']
  num_iters = args['num_iters']
  reg = args['regularization']
  unobs = args['unobs_weight']
  wt_type = args['wt_type']
  feature_wt_exp = args['feature_wt_exp']
  obs_wt = args['feature_wt_factor']

  tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # generate model
  input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse,
                                                                dim,
                                                                reg,
                                                                unobs,
                                                                args['weights'],
                                                                wt_type,
                                                                feature_wt_exp,
                                                                obs_wt)

  # factorize matrix
  session = wals.simple_train(model, input_tensor, num_iters)

  tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # evaluate output factor matrices
  output_row = row_factor.eval(session=session)
  output_col = col_factor.eval(session=session)

  # close the training session now that we've evaluated the output
  session.close()

  return output_row, output_col


def save_model(args, user_map, item_map, row_factor, col_factor):
  """Save the user map, item map, row factor and column factor matrices in numpy format.

  These matrices together constitute the "recommendation model."

  Args:
    args:         input args to training job
    user_map:     user map numpy array
    item_map:     item map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
  """
  model_dir = os.path.join(args['output_dir'], 'model')

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args['job_name'])

  os.makedirs(model_dir)
  np.save(os.path.join(model_dir, 'user'), user_map)
  np.save(os.path.join(model_dir, 'item'), item_map)
  np.save(os.path.join(model_dir, 'row'), row_factor)
  np.save(os.path.join(model_dir, 'col'), col_factor)

  with open(os.path.join(model_dir, 'user.json'), 'w') as outfile:
    json.dump(user_map, outfile,cls=NumpyArrayEncoder)

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

def save_model_json(args, user_map, item_map, row_factor, col_factor):
  """Save the user map, item map, row factor and column factor matrices in json format.

  These matrices together constitute the "recommendation model."

  Args:
    args:         input args to training job
    user_map:     user map numpy array
    item_map:     item map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
  """
  model_dir = os.path.join(args['output_dir'], 'model')

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args['job_name'])

  os.makedirs(model_dir)
  np.save(os.path.join(model_dir, 'user'), user_map)
  np.save(os.path.join(model_dir, 'item'), item_map)
  np.save(os.path.join(model_dir, 'row'), row_factor)
  np.save(os.path.join(model_dir, 'col'), col_factor)

  with open(os.path.join(model_dir, 'user.json'), 'w') as outfile:
    json.dump(user_map, outfile,cls=NumpyArrayEncoder)
  with open(os.path.join(model_dir, 'item.json'), 'w') as outfile:
    json.dump(item_map, outfile,cls=NumpyArrayEncoder)
  with open(os.path.join(model_dir, 'row.json'), 'w') as outfile:
    json.dump(row_factor, outfile,cls=NumpyArrayEncoder)
  with open(os.path.join(model_dir, 'col.json'), 'w') as outfile:
    json.dump(col_factor, outfile,cls=NumpyArrayEncoder)

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

def save_user_items_w(args, user_items_w):
  """Save the user map, item map, row factor and column factor matrices in numpy format.

  """
  model_dir = os.path.join(args['output_dir'], 'model')

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args['job_name'])

  # os.makedirs(model_dir)
  
  user_items_w.to_json(os.path.join(model_dir, 'user_item_w.json'),orient='records')

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
  """Generate recommendations for a user.

  Args:
    user_idx: the row index of the user in the ratings matrix,

    user_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations)

    row_factor: the row factors of the recommendation model
    col_factor: the column factors of the recommendation model

    k: number of recommendations requested

  Returns:
    list of k item indexes with the predicted highest rating, excluding
    those that the user has already rated
  """

  # bounds checking for args
  assert (row_factor.shape[0] - len(user_rated)) >= k

  # retrieve user factor
  user_f = row_factor[user_idx]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(user_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(user_rated)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in user_rated]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items
