---
layout: post
title:  "Kalman Filter Initialization - The Stationary Case"
date:   2014-06-23 00:06:18 -07:00
categories: python
permalink: /topics/kalman_init_stationary.html
redirect_from: /posts/kalman_init_stationary.html
notebook: kalman_filter_initialization_stationary
---

**Note**: the pull request described below has been merged into Scipy, so the
timings below are no longer accurate - in particular,
`scipy.linalg.solve_discrete_lyapunov` is now much faster for large matrix
inputs due to the use of one of the bilinear transformations described below.
This page is kept as-is for a historical look at how slow a naive approach can
be.
