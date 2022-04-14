---
layout: post
title: "Building a Recommender System, Part 2"
tagline: "I think I have a good shot at winning the Netflix prize"
tags: [recommender, tensorflow]
author: Matthew Mahowald
mathjax: true
---

In a previous post, we looked at neighborhood-based methods for building
recommender systems. This post explores an alternative technique for collaborative
filtering using latent factor models. The technique we'll use naturally generalizes
to deep learning approaches (such as autoencoders), so we'll also implement
our approach using Tensorflow and Keras.

The Dataset
-----------

We'll re-use the same MovieLens dataset for this post that we worked on last
time for our collaborative filtering model. [GroupLens](https://grouplens.org)
has [made the dataset available here](https://grouplens.org/datasets/movielens/).

First, let's load in this data:

```python
import pandas as pd
import numpy as np

np.random.seed(42)

ratings = pd.read_csv(RATING_DATA_FILE,
            sep='::',
            engine='python',
            encoding='latin-1',
            names=['userid', 'movieid', 'rating', 'timestamp'])

movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE),
            sep='::',
            engine='python',
            encoding='latin-1',
            names=['movieid', 'title', 'genre']).set_index("movieid")
```

Let's take a quick look at the top 20 most-viewed files:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genre</th>
    </tr>
    <tr>
      <th>movieid</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2858</th>
      <td>American Beauty (1999)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>260</th>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>Action|Adventure|Fantasy|Sci-Fi</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
      <td>Action|Adventure|Drama|Sci-Fi|War</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>Star Wars: Episode VI - Return of the Jedi (1983)</td>
      <td>Action|Adventure|Romance|Sci-Fi|War</td>
    </tr>
    <tr>
      <th>480</th>
      <td>Jurassic Park (1993)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>Saving Private Ryan (1998)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>589</th>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>2571</th>
      <td>Matrix, The (1999)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>Back to the Future (1985)</td>
      <td>Comedy|Sci-Fi</td>
    </tr>
    <tr>
      <th>593</th>
      <td>Silence of the Lambs, The (1991)</td>
      <td>Drama|Thriller</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>Men in Black (1997)</td>
      <td>Action|Adventure|Comedy|Sci-Fi</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>Raiders of the Lost Ark (1981)</td>
      <td>Action|Adventure</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Fargo (1996)</td>
      <td>Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>2762</th>
      <td>Sixth Sense, The (1999)</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Braveheart (1995)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>2396</th>
      <td>Shakespeare in Love (1998)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>Princess Bride, The (1987)</td>
      <td>Action|Adventure|Comedy|Romance</td>
    </tr>
    <tr>
      <th>527</th>
      <td>Schindler's List (1993)</td>
      <td>Drama|War</td>
    </tr>
    <tr>
      <th>1617</th>
      <td>L.A. Confidential (1997)</td>
      <td>Crime|Film-Noir|Mystery|Thriller</td>
    </tr>
    <tr>
      <th>1265</th>
      <td>Groundhog Day (1993)</td>
      <td>Comedy|Romance</td>
    </tr>
  </tbody>
</table>
</div>

Preprocessing
-------------

Collaborative filtering models typically work best when each item has a
decent number of ratings. Let's restrict to only the 500 most popular films
(as determined by number of ratings). We'll also reindex by `movieid` and `userid`:

```python
rating_counts = (
  ratings
  .groupby("movieid")["rating"]
  .count()
  .sort_values(ascending=False)
)

# only the 500 most popular movies
pop_ratings = ratings[ratings["movieid"].isin((rating_counts).index[0:500])]
pop_ratings = pop_ratings.set_index(["movieid", "userid"])
```

Next, [as mentioned in the previous post](../../04/25/recommender-systems.html), we should normalize our rating data.
We create an adjusted rating by subtracting off the overall mean rating, the
mean rating for each item, and then the mean rating for each user.

This produces a "preference rating" $\tilde{r}_{u,i}$ defined by

$$
\tilde{r}_{u,i} := r_{u,i} - \bar{r} - \bar{r}_{i} - \bar{r}_{u}
$$

The intuition for $\tilde{r}$
is that $\tilde{r} = 0$ means that user
$u$'s rating for item $i$ is exactly what we would guess if all we knew was
the average overall ratings, item ratings, and user ratings. Any values above or
below 0 indicate deviations in preference from this baseline. To distinguish
$\tilde{r}$ from the raw rating $r$, I'll refer to the former as
the user's _preference_ for item $i$ and the latter as the user's _rating_ of
item $i$.

Let's build the preference data using ratings for the 500 most popular films:

```python
prefs = pop_ratings["rating"]

mean_0 = pop_ratings["rating"].mean()
prefs = prefs - mean_0

mean_i = prefs.groupby("movieid").mean()
prefs = prefs - mean_i

mean_u = prefs.groupby("userid").mean()
prefs = prefs - mean_u


pref_matrix = (
  prefs
  .reset_index()[["userid", "movieid", "rating"]]
  .pivot(index="userid", columns="movieid", values="rating")
)
```

The output of this block of code is two objects: `prefs`, which is a dataframe
of preferences indexed by `movieid` and `userid`; and `pref_matrix`, which is
a matrix whose $(i,j)$th entry corresponds to the rating user $i$ gives movie $j$
(i.e. the columns are movies and each row is a user). In cases where the user
hasn't rated the item, this matrix will have a `NaN`.

The maximum and minimum preferences in this data are 3.923 and -4.643, respectively.
Next, we'll build an actual model.

Latent-factor collaborative filtering
-------------------------------------

At this stage, we've constructed a matrix $P$ (called `pref_matrix` in the Python
code above). The idea behind latent-factor collaborative filtering models is that
each user's preferences can be predicted by a small number of latent factors (usually
much smaller than the overall number of items available):

$$
\tilde{r}_{u,i} \approx f_{i}(\lambda_{1}(u), \lambda_{2}(u), \ldots, \lambda_{n}(u))
$$

Latent factor models thus require answering two related questions:

1. For a given user $u$, what are the corresponding latent factors $\lambda_{k}(u)$?
2. For a given collection of latent factors, what is the function $f_{i}$, i.e.,
   what is the relationship between the latent factors and a user's preferences
   for each item?

One approach to this problem is to attempt to solve for
both the $f_{i}$'s and $\lambda_{k}$'s by making the simplifying assumption that
each of these functions is linear:

$$
\lambda_{k}(u) = \sum_{i} a_{i} \tilde{r}_{u,i}
$$

$$
f_{i} = \sum_{k} b_{k} \lambda_{k}
$$

Taken over all items and users, this can be re-written as a linear algebra problem
problem: find matrices $F$ and $\Lambda$ such that

$$ P \approx F  \Lambda  P, $$

where $P$ is the matrix of preferences, $\Lambda$ is the linear transformation
that projects a user's preferences onto latent variable space, and $F$ is the
linear transformation that reconstructs the user's ratings from that user's
representation in latent variable space.

The product $F \Lambda$ will be a square matrix. However, by choosing a number of
latent variables strictly less than the number of items, this product will
necessarily not be full rank. In essence, we are solving for $F$ and $\Lambda$
such that the product $F \Lambda$ best approximates the identity transformation _on the
preferences matrix_ $P$. Our intuition (and hope) is that this will reconstruct
accurate preferences for each user. (We will tune our loss function to ensure
that this is in fact the case.)

Model implementation
--------------------

As advertised, we'll be building our model in Keras + Tensorflow so that we're
well-positioned for any future generalization to deep learning approaches. This
is also a natural approach to the type of problem we're solving: the expression

$$
P \approx F \Lambda P
$$

can be thought of as describing a two-layer dense neural network whose layers are
defined by $F$ and $\Lambda$ and whose activation function is just the identity
map (i.e. the function $\sigma(x) = x$).

First, let's import the packages we'll need and set the encoding dimension (the
number of latent variables) we want for this model.

```python
import tensorflow as tf

from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model as keras_load_model
from keras import losses
from keras.callbacks import EarlyStopping

ENCODING_DIM = 25
ITEM_COUNT = 500
```

Next, define the model itself as a composition of an "encoding" layer (projection
onto latent variable space) and a "decoding" layer (recovery of preferences from
latent variable representation). The recommender model itself is just a composition
of these two layers.

```python
# ~~~ build recommender ~~~ #
input_layer = Input(shape=(ITEM_COUNT, ))
# compress to low dimension
encoded = Dense(ENCODING_DIM, 
                activation="linear", 
                use_bias=False)(input_layer)
# blow up to large dimension
decoded = Dense(ITEM_COUNT, 
                activation="linear", 
                use_bias=False)(encoded)       

# define subsets of the model:
# 1. the recommender itself
recommender = Model(input_layer, decoded)

# 2. the encoder
encoder = Model(input_layer, encoded)

# 3. the decoder
encoded_input = Input(shape=(ENCODING_DIM, ))
decoder = Model(encoded_input, 
          recommender.layers[-1](encoded_input))
```

Custom loss functions
---------------------

At this point, we could train our model directly to just reproduce its
inputs (this is essentially a very simple autoencoder). However, we're actually
interested in picking $F$ and $\Lambda$ that correctly fill in _missing_ values.
We can do this through a careful application of masking and a custom loss function.

Recall that `prefs_matrix` currently consists largely of NaNs---in fact, there's
only one zero value in the whole dataset:

```python
prefs[prefs == 0]
# movieid  userid
# 2664     2204      0.0
```

In `prefs_matrix`, we can fill any missing values with zeros. This is a reasonable
choice because we've already performed some normalization of the ratings, so 0
represents our naive guess for a user's preference for a given item. Then, to create
training data, use `prefs_matrix` as the target and selectively mask nonzero
elements in `prefs_matrix` to create the input ("forgetting" that particular
user-item preference). We can then build a loss function which strongly penalizes
incorrectly guessing the "forgotten" values, i.e., one which is trained to
construct novel ratings from known ratings. Here's our function:

```python
def lambda_mse(frac=0.8):
    """
    Specialized loss function for recommender model.

    :param frac: Proportion of weight to give to novel ratings.
    :return: A loss function for use in a Lambda layer.
    """
    def lossfunc(xarray):
        x_in, y_true, y_pred = xarray
        zeros = tf.zeros_like(y_true)

        novel_mask = tf.not_equal(x_in, y_true)
        known_mask = tf.not_equal(x_in, zeros)

        y_true_1 = tf.boolean_mask(y_true, novel_mask)
        y_pred_1 = tf.boolean_mask(y_pred, novel_mask)

        y_true_2 = tf.boolean_mask(y_true, known_mask)
        y_pred_2 = tf.boolean_mask(y_pred, known_mask)

        unknown_loss = losses.mean_squared_error(y_true_1, y_pred_1)
        known_loss = losses.mean_squared_error(y_true_2, y_pred_2)

        # remove nans
        unknown_loss = tf.where(tf.is_nan(unknown_loss), 0.0, unknown_loss)

        return frac*unknown_loss + (1.0 - frac)*known_loss
    return lossfunc
```

By default, the loss this returns is a 20%-80% weighted sum of the overall MSE and the
MSE of just the missing ratings. This loss function requires the input (with missing
preferences), the predicted preferences, and the true preferences.

At least as of the date of this post, Keras and TensorFlow don't currently support
custom loss functions with three inputs (other frameworks, such as PyTorch, do). We can
get around this fact by introducing a "dummy" loss function and a simple wrapper
model. Loss functions in Keras require only two inputs, so this dummy function
will ignore the "true" values.

```python
def final_loss(y_true, y_pred):
    """
    Dummy loss function for wrapper model.
    :param y_true: true value (not used, but required by Keras)
    :param y_pred: predicted value
    :return: y_pred
    """
    return y_pred
```

Next, our wrapper model. The idea here is to use a lambda layer ('`loss`') to apply our
custom loss function (`'lambda_mse'`), and then use our custom loss function for
the actual optimization. Using Keras's functional API makes it very easy to
wrap the recommender we already defined with this simple wrapper model.

```python
original_inputs = recommender.input
y_true_inputs = Input(shape=(ITEM_COUNT, ))
original_outputs = recommender.output
# give 80% of the weight to guessing the missings, 
# 20% to reproducing the knowns
loss = Lambda(lambda_mse(0.8))(
  [original_inputs, y_true_inputs, original_outputs])

wrapper_model = Model(inputs=[original_inputs, y_true_inputs], 
                      outputs=[loss])
wrapper_model.compile(optimizer='adadelta', loss=final_loss)
```

Training
--------

To generate training data for our model, we'll start with the preferences matrix
`pref_matrix` and randomly mask (i.e. set to 0) a certain fraction of the known ratings
for each user. Structuring this as a generator allows us to make an essentially
unlimited collection of training data (though in each case, the output is constrained
to be drawn from the same fixed set of known ratings). Here's the generator function:

```python
def generate(pref_matrix, batch_size=64, mask_fraction=0.2):
    """
    Generate training triplets from this dataset.

    :param batch_size: Size of each training data batch.
    :param mask_fraction: Fraction of ratings in training
        data input to mask. 0.2 = hide 20% of input ratings.
    :param repeat: Steps between shuffles.
    :return: A generator that returns tuples of the form 
        ([X, y], zeros) where X, y, and zeros all have
        shape[0] = batch_size. X, y are training inputs for 
        the recommender.
    """

    def select_and_mask(frac):
        def applier(row):
            row = row.copy()
            idx = np.where(row != 0)[0]
            if len(idx) > 0:
                masked = np.random.choice(idx, 
                    size=(int)(frac*len(idx)), replace=False)
                row[masked] = 0
            return row
        return applier

    indices = np.arange(pref_matrix.shape[0])
    batches_per_epoch = int(np.floor(len(indices)/batch_size))
    while True:
        np.random.shuffle(indices)

        for batch in range(0, batches_per_epoch):
            idx = indices[batch*batch_size:(batch+1)*batch_size]

            y = np.array(pref_matrix[idx,:])
            X = np.apply_along_axis(
              select_and_mask(frac=mask_fraction),
              axis=1, arr=y)

            yield [X, y], np.zeros(batch_size)
```

Let's check that this generator's masking functionality is working correctly:

```python
[X, y], _ = next(generate(pref_matrix.fillna(0).values))
len(X[X != 0])/len(y[y != 0])
# returns 0.8040994014148377
```

To complete the story, we'll define a training function that calls this
generator and allows us to set a few other parameters (number of epochs,
early stopping, etc):

```python
def fit(wrapper_model,
        pref_matrix, 
        batch_size=64, 
        mask_fraction=0.2, 
        epochs=1, 
        verbose=1, 
        patience=0):
    stopper = EarlyStopping(monitor="loss", 
        min_delta=0.00001, 
        patience=patience, 
        verbose=verbose)
    batches_per_epoch = int(np.floor(pref_matrix.shape[0]/batch_size))

    generator = generate(pref_matrix, batch_size, mask_fraction)

    history = wrapper_model.fit_generator(
        generator,
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        callbacks = [stopper] if patience > 0 else []
    )

    return history
```

Recall that $\Lambda$ and $F$ are $500 \times 25$ and $25 \times 500$ dimensional
matrices, respectively, so this model has $2 \times 25 \times 500 = 25000$ parameters.
A good rule of thumb with linear models is to have at least 10 observations per parameter, meaning we'd
like to see 250,000 individual user ratings vectors during training. We don't
have nearly enough users for that, though, so for this tutorial, we'll skimp by
quite a bit---let's settle for a maximum of 12,500 observations (stopping the
model earlier if loss doesn't improve).

```python
# stop after 3 epochs with no improvement
fit(wrapper_model, 
    pref_matrix.fillna(0).values, 
    batch_size=125, epochs=100, patience=3)
# Loss of 0.6321
```

The output of this training process (at least on my machine) gives a loss of
0.6321, which means that on average we're within about 0.7901 units of a user's
true preference when we haven't seen it before (recall that this loss is 80% from
unknown preferences, and 20% from the knowns). Preferences in our data range between
-4.64 and 3.92, so this is not too shabby!

Predicting ratings
------------------

To generate a prediction with our model, we have to call the `recommender` model
we trained earlier after normalizing the ratings along the various dimensions.
Let's assume that the input to our predict function will be a dataframe indexed
by (`movieid`, `userid`), and with a single column named `"rating"`.

```python
def predict(ratings, recommender, mean_0, mean_i, movies):
    # add a dummy user that's seen all the movies so when we generate
    # the ratings matrix, it has the appropriate columns
    dummy_user = movies.reset_index()[["movieid"]].copy()
    dummy_user["userid"] = -99999
    dummy_user["rating"] = 0
    dummy_user = dummy_user.set_index(["movieid", "userid"])

    ratings = ratings["rating"]

    ratings = ratings - mean_0
    ratings = ratings - mean_i
    mean_u = ratings.groupby("userid").mean()
    ratings = ratings - mean_u

    ratings = ratings.append(dummy_user["rating"])

    pref_mat = (ratings
                .reset_index()[["userid", "movieid", "rating"]]
                .pivot(index="userid", 
                       columns="movieid", 
                       values="rating")
              )
    X = pref_mat.fillna(0).values
    y = recommender.predict(X)

    output = pd.DataFrame(y, index=pref_mat.index, columns=pref_mat.columns)
    output = output.iloc[1:] # drop the bad user

    output = output.add(mean_u, axis=0)
    output = output.add(mean_i, axis=1)
    output = output.add(mean_0)

    return output
```

Let's test it out! Here's some sample ratings for a single fake user,
who really likes Star Wars and Jurassic Park and doesn't like much else:

```python
sample_ratings = pd.DataFrame([
    {"userid": 1, "movieid": 2858, "rating": 1}, # american beauty
    {"userid": 1, "movieid": 260, "rating": 5},  # star wars
    {"userid": 1, "movieid": 480, "rating": 5},  # jurassic park
    {"userid": 1, "movieid": 593, "rating": 2},  # silence of the lambs
    {"userid": 1, "movieid": 2396, "rating": 2}, # shakespeare in love
    {"userid": 1, "movieid": 1197, "rating": 5}  # princess bride
]).set_index(["movieid", "userid"])

# predict and print the top 10 ratings for this user
y = predict(sample_ratings, 
        recommender,
        mean_0, mean_i, 
        movies.loc[(rating_counts).index[0:500]]).transpose()
preds = y.sort_values(by=1, ascending=False).head(10)

preds["title"] = movies.loc[preds.index]["title"]
preds
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>userid</th>
      <th>1</th>
      <th>title</th>
    </tr>
    <tr>
      <th>movieid</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>260</th>
      <td>4.008329</td>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>3.942005</td>
      <td>Raiders of the Lost Ark (1981)</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>3.860034</td>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>3.716259</td>
      <td>Wrong Trousers, The (1993)</td>
    </tr>
    <tr>
      <th>904</th>
      <td>3.683811</td>
      <td>Rear Window (1954)</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>3.654374</td>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
    </tr>
    <tr>
      <th>913</th>
      <td>3.639756</td>
      <td>Maltese Falcon, The (1941)</td>
    </tr>
    <tr>
      <th>318</th>
      <td>3.637150</td>
      <td>Shawshank Redemption, The (1994)</td>
    </tr>
    <tr>
      <th>745</th>
      <td>3.619762</td>
      <td>Close Shave, A (1995)</td>
    </tr>
    <tr>
      <th>908</th>
      <td>3.608473</td>
      <td>North by Northwest (1959)</td>
    </tr>
  </tbody>
</table>
</div>

Interestingly, even though the user gave _Star Wars_ a 5 as input, the model only
predicts a rating of 4.08 for _Star Wars_. But it does recommend _the Empire
Strikes Back_ and _Raiders of the Lost Ark_, which seem like reasonable recommendations
for those preferences.

Now let's reverse this user's ratings for Star Wars and Jurassic Park, and see how
the ratings change:

```python
sample_ratings2 = pd.DataFrame([
    {"userid": 1, "movieid": 2858, "rating": 5}, # american beauty
    {"userid": 1, "movieid": 260, "rating": 1},  # star wars
    {"userid": 1, "movieid": 480, "rating": 1},  # jurassic park
    {"userid": 1, "movieid": 593, "rating": 1},  # silence of the lambs
    {"userid": 1, "movieid": 2396, "rating": 5}, # shakespeare in love
    {"userid": 1, "movieid": 1197, "rating": 5}  # princess bride
]).set_index(["movieid", "userid"])

y = predict(sample_ratings2, 
            recommender, 
            mean_0, mean_i, 
            movies.loc[(rating_counts).index[0:500]]
            ).transpose()
preds = y.sort_values(by=1, ascending=False).head(10)

preds["title"] = movies.loc[preds.index]["title"]
preds
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>userid</th>
      <th>1</th>
      <th>title</th>
    </tr>
    <tr>
      <th>movieid</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019</th>
      <td>3.532214</td>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3.489284</td>
      <td>Usual Suspects, The (1995)</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>3.480124</td>
      <td>American Beauty (1999)</td>
    </tr>
    <tr>
      <th>745</th>
      <td>3.466157</td>
      <td>Close Shave, A (1995)</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>3.415981</td>
      <td>Wrong Trousers, The (1993)</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>3.415527</td>
      <td>Princess Bride, The (1987)</td>
    </tr>
    <tr>
      <th>527</th>
      <td>3.386785</td>
      <td>Schindler's List (1993)</td>
    </tr>
    <tr>
      <th>750</th>
      <td>3.342154</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>3.338330</td>
      <td>Chinatown (1974)</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>3.335204</td>
      <td>To Kill a Mockingbird (1962)</td>
    </tr>
  </tbody>
</table>
</div>

Note that _Seven Samurai_ features prominently in both lists. In fact, _Seven Samurai_
has the highest average rating of any film in this dataset (at 4.56), and looking at
the top 20 or top 50 recommended films for both users has even more common films
showing up that happen to be very highly rated overall.

Conclusions and further reading
-------------------------------

The latent factor representation we've built can also be thought of as defining
an embedding of _items_ into some lower-dimensional space, as opposed to an
embedding of _users_. This lets us do some interesting things---for example, we
can compare distances between each item's vector representation to understand
how similar or different two films are. Let's compare _Star Wars_ against
_The Empire Strikes Back_ and _American Beauty_:

```python
starwars = decoder.get_weights()[0][:,33]
esb = decoder.get_weights()[0][:,144]
americanbeauty = decoder.get_weights()[0][:,401]
```

Note that 33 is the column index corresponding to _Star Wars_ (different from its
`movieid` of 260), 144 is the column index corresponding to _Empire Strikes Back_,
and 401 is the column index of _American Beauty_.

```python
np.sqrt(((starwars - esb)**2).sum())
# 0.209578

np.sqrt(((starwars - americanbeauty)**2).sum())
# 0.613659
```

Comparing the distances, we see that with a distance of 0.209578, _Star Wars_ and
_Empire Strikes Back_ are much closer in latent factor space than _Star Wars_
and _American Beauty_ are.

With a little bit of further work, it's also possible to answer other questions
in latent factor space like "which film is least similar to _Star Wars_?"

Variations on this type of technique lead to autoencoder-based recommender systems.
For futher reading, there's also a family of related models known as matrix
factorization models, which can incorporate both item and user features as well
as the raw ratings.

_A version of this post originally appeared [on KDNuggets](https://www.kdnuggets.com/2019/07/building-recommender-system-part-2.html)._