---
layout: page
title: Research
icon: calculator
index: 10
permalink: /research.html
---
{::options parse_block_html="true" /}

<section class="lead-section compact">

# Research

My research focuses on rational inattention and applied time series
econometrics.

- [Research](#research)
  - [Published papers](#published-papers)
    - [Choosing what to pay attention to (2022)](#choosing-what-to-pay-attention-to-2021)
    - [Forecasting US inflation in real time (2022)](#forecasting-us-inflation-in-real-time-2021)
    - [Bayesian Estimation and Forecasting of Time Series in statsmodels (2022)](#bayesian-estimation-and-forecasting-of-time-series-in-statsmodels-2022)
    - [SciPy 1.0: fundamental algorithms for scientific computing in Python (2020)](#scipy-10-fundamental-algorithms-for-scientific-computing-in-python-2020)
  - [Working papers](#working-papers)
    - [Mechanics of static quadratic Gaussian rational inattention tracking problems (2018)](#mechanics-of-static-quadratic-gaussian-rational-inattention-tracking-problems-2018)
    - [Mechanics of linear quadratic Gaussian rational inattention tracking problems (2017)](#mechanics-of-linear-quadratic-gaussian-rational-inattention-tracking-problems-2017)
    - [Estimating time series models by state space methods in Python: Statsmodels (2015)](#estimating-time-series-models-by-state-space-methods-in-python-statsmodels-2015)
  - [Other research](#other-research)
    - [Index of Common Inflation Expectations (2020)](#index-of-common-inflation-expectations-2020)

</section>

<div style="padding:0px 45px;padding-top:10px;margin-top:0px;border-top:1px solid #efefef;">

## Published papers

</div>

<section id="fulton_choosing_2021">

### Choosing what to pay attention to (2022)

*Theoretical Economics, 2022*

This paper studies static rational inattention problems with multiple
actions and multiple shocks. We solve for the optimal signals chosen by
agents and provide tools to interpret information processing. By relaxing
restrictive assumptions previously used to gain tractability, we allow
agents more latitude to choose what to pay attention to. Our applications
examine the pricing problem of a monopolist who sells in multiple markets
and the portfolio problem of an investor who can invest in multiple assets.
The more general models that our methods allow us to solve yield new
results. We show conditions under which the multimarket monopolist would
optimally choose a uniform pricing strategy, and we show how optimal
information processing by rationally inattentive investors can be
interpreted as learning about the Sharpe ratio of a diversified portfolio.

- <a href="https://onlinelibrary.wiley.com/doi/full/10.3982/TE3850" onclick="trackOutboundLink('https://onlinelibrary.wiley.com/doi/full/10.3982/TE3850'); return false;">Published version (PDF)</a>
- <a href="https://econtheory.org/ojs/index.php/te/article/viewForthcomingFile/3850/30749/1" onclick="trackOutboundLink('https://econtheory.org/ojs/index.php/te/article/viewForthcomingFile/3850/30749/1'); return false;">Working paper version (PDF)</a>

</section>

<section id="fulton_forecasting_2021">

### Forecasting US inflation in real time (2022)

*Econometrics, 2022*, with [Kirstin Hubrich](https://sites.google.com/site/kirstinhubrichwebpage/)

We analyze real-time forecasts of US inflation over 1999Q3–2019Q4 and subsamples, investigating whether and how forecast accuracy and robustness can be improved with additional information such as expert judgment, additional macroeconomic variables, and forecast combination. The forecasts include those from the Federal Reserve Board’s Tealbook, the Survey of Professional Forecasters, dynamic models, and combinations thereof. While simple models remain hard to beat, additional information does improve forecasts, especially after 2009. Notably, forecast combination improves forecast accuracy over simpler models and robustifies against bad forecasts; aggregating forecasts of inflation’s components can improve performance compared to forecasting the aggregate directly; and judgmental forecasts, which may incorporate larger and more timely datasets in conjunction with model-based forecasts, improve forecasts at short horizons.

- <a href="https://www.mdpi.com/2225-1146/9/4/36" onclick="trackOutboundLink('https://www.mdpi.com/2225-1146/9/4/36'); return false;">Published version (PDF)</a>
- <a href="https://www.federalreserve.gov/econres/feds/files/2021014pap.pdf" onclick="trackOutboundLink('https://www.federalreserve.gov/econres/feds/files/2021014pap.pdf'); return false;">Working paper version (PDF)</a>

</section>

<section id="bayesian-estimation-and-forecasting-of-time-series-in-statsmodels-2022">

### Bayesian Estimation and Forecasting of Time Series in statsmodels (2022)

*Proceedings of the 21st Python in Science Conference, 2022*

Statsmodels, a Python library for statistical and econometric analysis, has traditionally focused on frequentist inference, including in its models for time series data. This paper introduces the powerful features for Bayesian inference of time series models that exist in statsmodels, with applications to model fitting, forecasting, time series decomposition, data simulation, and impulse response functions.

- <a href="https://conference.scipy.org/proceedings/scipy2022/chad_fulton.html" onclick="trackOutboundLink('https://conference.scipy.org/proceedings/scipy2022/chad_fulton.html'); return false;">Published version</a>
- <a href="https://github.com/ChadFulton/scipy2022-bayesian-time-series" onclick="trackOutboundLink('https://github.com/ChadFulton/scipy2022-bayesian-time-series'); return false;">Github Repository</a>

</section>

<section id="virtanen_scipy_2020">

### SciPy 1.0: fundamental algorithms for scientific computing in Python (2020)

*Nature methods, 2020*, with Pauli Virtanen, Ralf Gommers, and 108 others

SciPy is an open-source scientific computing library for the Python programming language. Since its initial release in 2001, SciPy has become a de facto standard for leveraging scientific algorithms in Python, with over 600 unique code contributors, thousands of dependent packages, over 100,000 dependent repositories and millions of downloads per year. In this work, we provide an overview of the capabilities and development practices of SciPy 1.0 and highlight some recent technical developments.

- <a href="https://www.nature.com/articles/s41592-019-0686-2?es_p=11046330" onclick="trackOutboundLink('https://www.nature.com/articles/s41592-019-0686-2?es_p=11046330'); return false;">Published version</a>

</section>

<div style="padding:0px 45px;">

## Working papers

</div>

<section id="fulton_static_ri_2017">

### Mechanics of static quadratic Gaussian rational inattention tracking problems (2018)

This paper presents a general framework for constructing and solving the
multivariate static linear quadratic Gaussian (LQG) rational inattention
tracking problem. We interpret the nature of the solution and the
implied action of the agent, and we construct representations that
formalize how the agent processes data. We apply our approach to a
price-setting problem and a portfolio choice problem - two popular
rational inattention models found in the literature for which
simplifying assumptions have thus far been required to produce a
tractable model. In contrast to prior results, which have been limited
to cases that restrict the number of underlying shocks or their
correlation structure, we present general solutions. In each case, we
show that imposing such restrictions impacts the form and interpretation
of solutions and implies suboptimal decision-making by agents.

- <a href="http://www.chadfulton.com/files/fulton_static_ri_2017.6f28145.pdf" onclick="trackOutboundLink('http://www.chadfulton.com/files/fulton_static_ri_2017.6f28145.pdf'); return false;">Working paper (PDF)</a>

</section>

<section id="fulton_mechanics_ri_2017">

### Mechanics of linear quadratic Gaussian rational inattention tracking problems (2017)

**Note**: *This is an previous version of the working paper [Mechanics of static quadratic Gaussian rational inattention tracking problems](#fulton_static_ri_2017), although it contains some sections not included there. In particular, it expands on the dynamic case and provides more detail on the equilibrium solution to the rational inattetion price-setting problem.*

This paper presents a general framework for constructing and solving the multivariate static linear quadratic Gaussian (LQG) rational inattention tracking problem. We interpret the nature of the solution and the implied action of the agent, and we construct representations that formalize how the agent processes data. We apply this infrastructure to the rational inattention price-setting problem, confirming the result that a conditional response to economics shocks is possible, but casting doubt on a common assumption made in the literature. We show that multiple equilibria and a social cost of increased attention can arise in these models. We consider the extension to the dynamic problem and provide an approximate solution method that achieves low approximation error for many applications found in the LQG rational inattention literature.

- <a href="https://www.federalreserve.gov/econres/feds/files/2017109pap.pdf" onclick="trackOutboundLink('https://www.federalreserve.gov/econres/feds/files/2017109pap.pdf'); return false;">Working paper (PDF)</a>

</section>

<section id="est-ssm-py">

### Estimating time series models by state space methods in Python: Statsmodels (2015)

This paper describes an object oriented approach to the estimation of time series models using state space methods and presents an implementation in the Python programming language. This approach at once allows for fast computation, a variety of out-of-the-box features, and easy extensibility. We show how to construct a custom state space model, retrieve filtered and smoothed estimates of the unobserved state, and perform parameter estimation using classical and Bayesian methods. The mapping from theory to implementation is presented explicitly and is illustrated at each step by the development of three example models: an ARMA(1,1) model, the local level model, and a simple real business cycle macroeconomic model. Finally, four fully implemented time series models are presented: SARIMAX, VARMAX, unobserved components, and dynamic factor models. These models can immediately be applied by users.

- [Working paper (PDF)]({{ "/files/fulton_statsmodels_2017_v1.pdf" | relative_url }})
- [Working paper (HTML)](https://chadfulton.github.io/fulton_statsmodels_2017/)
- [Github Repository](https://github.com/ChadFulton/fulton_statsmodels_2017)

</section>

<div style="padding:0px 45px;">

## Other research

</div>

<section id="ahn_index_2020">

### Index of Common Inflation Expectations (2020)

with [Hie Joo Ahn](https://sites.google.com/site/hiejooahn/)

This note develops a new index of common inflation expectations that summarizes the comovement of various inflation expectation indicators based on a dynamic factor model. This index suggests that inflation expectations were relatively stable between 1999 and 2012, and then experienced a downward shift that persisted, despite some fluctuations, at least through the beginning of the COVID-19 pandemic in early 2020. Since then it has successfully captured pandemic-driven concerns, first falling on fears of a prolonged recession, and then rising as the US economy has recovered and anxiety about inflation has grown.

- <a href="https://www.federalreserve.gov/econres/notes/feds-notes/index-of-common-inflation-expectations-20200902.htm" onclick="trackOutboundLink('https://www.federalreserve.gov/econres/notes/feds-notes/index-of-common-inflation-expectations-20200902.htm'); return false;">FEDS Note (HTML)</a>
- <a href="https://www.federalreserve.gov/econres/notes/feds-notes/research-data-series-index-of-common-inflation-expectations-20210305.htm" onclick="trackOutboundLink('https://www.federalreserve.gov/econres/notes/feds-notes/research-data-series-index-of-common-inflation-expectations-20210305.htm'); return false;">Estimated index (updated quarterly)</a>

</section>
