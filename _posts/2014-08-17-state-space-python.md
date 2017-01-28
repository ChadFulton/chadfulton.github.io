---
layout: post
title:  "State space modeling in Python"
date:   2014-08-17 11:58:04 -07:00
categories: python
permalink: /topics/state_space_python.html
redirect_from: /posts/state_space_python.html
notebook: state_space_python
related:
    - name: "Introduction to state space models"
      link: /topics/implementing_state_space.html
      description: an overview of state space models, their implementation in Python, and provides example code to estimate simple ARMA models.
    - name: "State space diagnostics"
      link: /topics/state_space_diagnostics.html
      description: a description of diagnostic statistics and output for state space models.
    - name: "Bayesian state space estimation via Metropolis-Hastings"
      link: /topics/state_space_mh.html
      description: a description of estimation of state space models via Metropolis-Hastings (Bayesian posterior simulation)
    - name: "Estimating an RBC model"
      link: /topics/estimating_rbc.html
      description: an example of setting up, solving, and estimating a simple RBC model using the `statespace` library in Statsmodels
---

**Update**: (June, 2016) The notebook has been updated to include recent changes to the state space library.

**Update**: (February, 2015) The [pull request](https://github.com/statsmodels/statsmodels/pull/2250)
has been merged, and state space models will be included in 
Statsmodels beginning with version 0.7. The code, below, has
been updated from the original post to reflect the current
design.
