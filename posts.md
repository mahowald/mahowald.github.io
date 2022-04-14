---
layout: default
title: All Posts
permalink: /posts/
---

<h1 class="title">All Posts</h1>

Here you can read a selection of my finest words, listed in reverse chronological order!

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <div class="entry">
      <h1 class="post"><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>
          <div> {{ post.tagline }} </div>
          <div class="date">
              {{ post.date | date: "%B %e, %Y" }}
          </div>
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>