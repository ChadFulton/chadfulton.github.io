---
layout: post
title:  "Implementing and estimating an ARMA(1, 1) state space model"
date:   2017-01-28 22:25:11 -07:00
categories: time-series python statsmodels state-space maximum-likelihood bayesian metropolis-hastings gibbs-sampling arima sarimax
permalink: /topics/arma11_cpi_inflation.html
notebook: arma11_cpi_inflation
---

This notebook collects the full example implementing and estimating (via maximum likelihood, Metropolis-Hastings, and Gibbs Sampling) a specific autoregressive integrated moving average (ARIMA) model, from my working paper [Estimating time series models by state space methods in Python: Statsmodels]({{ "/research.html#est-ssm-py" | relative_url }}).