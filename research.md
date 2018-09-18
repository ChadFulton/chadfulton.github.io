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

## Working papers

- [Mechanics of static quadratic Gaussian rational inattention tracking problems](#fulton_static_ri_2017)
- [Mechanics of linear quadratic Gaussian rational inattention tracking problems](#fulton_mechanics_ri_2017)
- [Estimating time series models by state space methods in Python: Statsmodels](#est-ssm-py)

## Selected presentations

- [Estimating time series models by state space methods in Python: Statsmodels](#est-ssm-py) (Presentation at Securities and Exchange Commission)

</section>

<section id="fulton_static_ri_2017">

## Mechanics of static quadratic Gaussian rational inattention tracking problems

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

## Mechanics of linear quadratic Gaussian rational inattention tracking problems

**Note**: *This is an previous version of the working paper [Mechanics of static quadratic Gaussian rational inattention tracking problems](#fulton_static_ri_2017), although it contains some sections not included there. In particular, it expands on the dynamic case and provides more detail on the equilibrium solution to the rational inattetion price-setting problem.*

This paper presents a general framework for constructing and solving the multivariate static linear quadratic Gaussian (LQG) rational inattention tracking problem. We interpret the nature of the solution and the implied action of the agent, and we construct representations that formalize how the agent processes data. We apply this infrastructure to the rational inattention price-setting problem, confirming the result that a conditional response to economics shocks is possible, but casting doubt on a common assumption made in the literature. We show that multiple equilibria and a social cost of increased attention can arise in these models. We consider the extension to the dynamic problem and provide an approximate solution method that achieves low approximation error for many applications found in the LQG rational inattention literature.

- <a href="https://www.federalreserve.gov/econres/feds/files/2017109pap.pdf" onclick="trackOutboundLink('https://www.federalreserve.gov/econres/feds/files/2017109pap.pdf'); return false;">Working paper (PDF)</a>

</section>

<section id="est-ssm-py">

## Estimating time series models by state space methods in Python: Statsmodels

This paper describes an object oriented approach to the estimation of time series models using state space methods and presents an implementation in the Python programming language. This approach at once allows for fast computation, a variety of out-of-the-box features, and easy extensibility. We show how to construct a custom state space model, retrieve filtered and smoothed estimates of the unobserved state, and perform parameter estimation using classical and Bayesian methods. The mapping from theory to implementation is presented explicitly and is illustrated at each step by the development of three example models: an ARMA(1,1) model, the local level model, and a simple real business cycle macroeconomic model. Finally, four fully implemented time series models are presented: SARIMAX, VARMAX, unobserved components, and dynamic factor models. These models can immediately be applied by users.

- [Working paper (PDF)]({{ "/files/fulton_statsmodels_2017_v1.pdf" | relative_url }})
- [Working paper (HTML)](https://chadfulton.github.io/fulton_statsmodels_2017/)
- [Github Repository](https://github.com/ChadFulton/fulton_statsmodels_2017)

### Presentations

- [2018-09-15 - SEC]({{ "/files/fulton_statespace_sec_09-2018.pdf" | relative_url }})

</section>