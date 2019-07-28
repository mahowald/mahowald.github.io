---
layout: post
title: "Building a Recommender System, Part 1"
tags: [recommender]
author: Matthew Mahowald
mathjax: true
---

Recommender systems are one of the most prominent examples of machine learning
in the wild today. They determine what shows up in your Facebook news feed, in
what order products appear on Amazon, what videos are suggested in your Netflix
queue, as well as countless other examples. But what are recommender systems,
and how do they work? This post is the first in a series exploring some
common techniques for building recommender systems as well as their implementation.


What is a recommender system?
-----------------------------

A recommender system is an information filtering model that ranks or scores items
for users. There are generally two types of ranking methods:

* **Content-based filtering**, in which recommended items are based on item-to-item
  similarity and the user's explicit preferences; and
* **Collaborative filtering**, in which items are recommended to users based on
  the preferences of other users with similar transaction histories and
  characteristics.

The information used in collaborative filtering can be explicit, where users
provide ratings for each item, or implicit, where user preferences have to be
extracted from user behavior (purchases, views, etc). The most successful
recommender systems use hybrid approaches combining both filtering methods.

The MovieLens Datasets
----------------------

To make this discussion more concrete, let's focus on building recommender
systems using a specific example. [GroupLens](https://grouplens.org/), a research group at the University
of Minnesota, has generously made available [the MovieLens dataset](https://grouplens.org/datasets/movielens/).
This dataset consists of approximately 20 million user ratings applied to
27,000 movies by 138,000 users. In addition, the movies include genre and
date information. We'll use this dataset to build our model.

Simple Content-based Filtering
------------------------------

Let's build a simple recommender system that uses content-based filtering (i.e.
item similarity) to recommend movies for us to watch. First, load in the movie
dataset from MovieLens and multihot-encode the genre fields:

```python
import pandas as pd
import numpy as np

movies = pd.read_csv("movies.csv")
# dummy encode the genre
movies = movies.join(movies.genres.str.get_dummies("|"))
```

The `genres` feature consists of one or more pipe ("|") separated genres. The
last line above adds a column for each possible genre and puts a 1 in that entry
if the genre tag is present, or a 0 otherwise.

Let's generate some recommendations based on item similarity using these tags.
A very common similarity measure for categorical data (such as tags) is *cosine
similarity*. For any two items $i$ and $j$, the cosine similarity of $i$ and $j$
is simply the cosine of the angle between $i$ and $j$ where $i$ and $j$ are
interpreted as vectors in feature space. Recall that the cosine is obtained from
the inner product of these vectors:

$$
\cos \theta = \frac{i \cdot j}{||i|| ||j||}
$$

As a concrete example, consider the films $i := $ Toy Story (genre tags "Adventure",
"Animation", "Children", "Comedy", and "Fantasy") and $j := $ Jumanji (genre tags
"Adventure", "Children", and "Fantasy"). The dot product $i \cdot j$ is 3 (the
two films have three tags in common). $||i|| = \sqrt{5}$ and $||j|| = \sqrt{3}$,
so the cosine similarity between these two films is

$$
\cos \theta = \frac{3}{\sqrt{15}} \approx 0.775
$$

We can compute the cosine similarity for all of the items in our dataset:

```python
from sklearn.metrics.pairwise import cosine_similarity

# compute the cosine similarity
cos_sim = cosine_similarity(movies.iloc[:,3:])
```

The very first film in the dataset is Toy Story. Let's find out what the similar
films to Toy Story are:

```python
# Let's get the top 5 most similar films:
toystory_top5 = np.argsort(sim[0])[-5:][::-1]

# array([   0, 8219, 3568, 9430, 3000, 2809, 2355, 6194, 8927, 6948, 7760,
#       1706, 6486, 6260, 5490])

movies.iloc[toystory_top5]
```

MovieID|Genres
-------|------
Toy Story (1995)|Adventure,Animation,Children,Comedy,Fantasy
Turbo (2013)|Adventure,Animation,Children,Comedy,Fantasy
Monsters, Inc. (2001)|Adventure,Animation,Children,Comedy,Fantasy
Moana (2016)|Adventure,Animation,Children,Comedy,Fantasy
Emperor's New Groove, The (2000)|Adventure,Animation,Children,Comedy,Fantasy

The first five films all have exactly the same genre tags as Toy Story, and
hence a cosine similarity of 1. In fact, for the sample data used here, there
are thirteen films with similarity 1; the most similar film without identical
tags is 2006's "The Ant Bully", which has the additional genre tag "IMAX".

Simple Collaborative Filtering
------------------------------

Collaborative filtering recommends items based on what similar users liked.
Fortunately, in the MovieLens dataset, we have a wealth of user preference
information in the form of movie ratings: each user assigns one or more films
numeric ratings between 1 and 5 indicating how much they enjoyed the film.
We can view the problem of recommending items to the user as a _prediction_
task: given the user's ratings of other films, what is their likely rating of
the film in question?

One simple way to do this is to assign a similarity-weighted rating to each item
using other users' ratings:

$$
\hat{r}_{u,i} = \frac{\sum_{v\neq u} s(u,v) r_{v, i}}{\sum_{v \neq u} s(u,v)}
$$

where $\hat{r}_{u,i}$ is the predicted rating of item $i$ by user $u$, $s(u,v)$
is a measurement of similarity between users $u$ and $v$, and $r$ is the known
rating of item $i$ by user $v$.

For our user similarity measurement, we'll look at users' ratings of movies. Users
with similar ratings will be considered similar. To work with this rating data,
an important first step is to normalize our ratings. We'll do this in three steps:
first, we'll subtract the overall mean rating (across all films and users) so
that our adjusted ratings are centered at 0. Next, we'll do the same thing for
each film, to account for the mean ratings of a given film differing. Finally
we'll subtract off the mean rating for each user---this accounts for individual
variations (e.g. one user giving consistently higher ratings than another).

Mathematically, our adjusted rating $\tilde{r}_{u,i}$ is

$$
\tilde{r}_{u,i} := r_{u,i} - \bar{r} - \bar{r}_{i} - \bar{r}_{u}
$$

where $r_{u,i}$ is the base rating, $\bar{r}$ is the overall mean rating,
$\bar{r},i$ is the mean rating for item $i$ (after subtracting the overall mean),
and $\bar{r},u$ is the mean rating for user $u$ (after adjusting for the
overall mean rating and the item mean ratings). For convenience, I'll refer to
the adjusted rating $\tilde{r}$ as the _preference_ of user $u$ for item $i$.

Let's load in the ratings data and compute the adjusted ratings:

```python
ratings = pd.read_csv("ratings.csv")

mean_rating = ratings['rating'].mean() # compute mean rating

pref_matrix = (ratings[['userId', 'movieId', 'rating']]
               .pivot(index='userId', 
                      columns='movieId', 
                      values='rating')
            )

pref_matrix = pref_matrix - mean_rating # adjust by overall mean

item_mean_rating = pref_matrix.mean(axis=0)
pref_matrix = pref_matrix - item_mean_rating # adjust by item mean

user_mean_rating = pref_matrix.mean(axis=1)
pref_matrix = pref_matrix - user_mean_rating
```

At this point we can easily establish a reasonable baseline estimate for a
given user's rating of films they haven't seen:

```python
pref_matrix.fillna(0) + user_mean_rating + item_mean_rating + mean_rating
```

We can compute the distance to a particular user (in this case, user 0)
as follows:

```python
mat = pref_matrix.values
k = 0 # use the first user
np.nansum((mat - mat[k,:])**2,axis=1).reshape(-1,1)
```

It turns out that the nearest user is user 12 (with distance 0):

```python
np.nansum((mat - mat[0,:])**2,axis=1)[1:].argmin() # returns 11
# check it:
np.nansum(mat[12] - mat[0]) # returns 0.0
```

We find two films that user 12 has seen that user 0 has not:

```python
np.where(~np.isnan(mat[12]) & np.isnan(mat[0]) == True)
# returns (array([304, 596]),)

mat[12][[304, 596]]
# returns array([-2.13265214, -0.89476547])
```

Unfortunately, user 12 dislikes both of the films that user 0 hasn't seen yet!
We should continue our computation to account for all of the nearby users.

Concluding Remarks
------------------

The methods used in this post are *neighborhood-based*, and we've just seen above
a potential pitfall when generating recommendations based on neighbors: neighbors
may not actually recommend any items the user in question hasn't already seen.
Because of the need to compute pairwise distances, neighborhood-based methods
also tend to scale poorly as the number of users increases.

In part 2 of this series, we'll take a look at another approach for building
recommender systems, this time using *latent factor* methods. Latent factor
models avoid some of the pitfalls of the neighborhood-based methods described
here---but as we'll see, they come with some challenges of their own!

*This post [originally appeared on KDNuggets.](https://www.kdnuggets.com/2019/04/building-recommender-system.html)*
