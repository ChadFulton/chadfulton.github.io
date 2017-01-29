---
layout: post
title:  "Implementing and estimating a simple Real Business Cycle (RBC) model"
date:   2017-01-28 22:25:47 -07:00
categories: time-series python statsmodels state-space maximum-likelihood bayesian metropolis-hastings gibbs-sampling real-business-cycle dsge-model
permalink: /topics/simple_rbc.html
notebook: simple_rbc
related:
    - name: "Estimating a Real Business Cycle DSGE Model by Maximum Likelihood in Python"
      link: /topics/estimating_rbc.html
      description: a more complete look at formulating, solving, and estimating (by maximum likelihood) this same RBC model
---

This notebook collects the full example implementing and estimating (via maximum likelihood, Metropolis-Hastings, and Gibbs Sampling) a simple real business cycle model, from my working paper [Estimating time series models by state space methods in Python: Statsmodels]({{ "/research.html#est-ssm-py" | relative_url }}).