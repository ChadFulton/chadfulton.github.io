---
layout: page
title: Software
icon: terminal
permalink: /software.html
---
{::options parse_block_html="true" /}


<section id="code" class="lead-section compact">

# Economics and quantitative methods

<img class="ipython" src="/assets/images/ipython-gdp.png" />

Computational tools are an inescapable component of modern economic research. I
have contributed to a number of open-source software projects to improve
freely available time series econometrics software. These contributions
include:

- [Estimation of state space models (Statsmodels)](#state-space)
- [Estimation of Markov switching models (Statsmodels)](#markov-switching)
- [Wishart random variables and sampling (Scipy)](#wishart)

<br /><br />

</section>

<section id="state-space" class="state-space">

<div class="timeline">
<table class="timeline" cellpadding="0" cellspacing="0">
    <tbody>
        <tr class="dates">
            <td class="y14m05">
                <span>
                    <a href="http://www.statsmodels.org/dev/statespace.html"> Kalman Filter</a>
                </span>
            </td>
            <td class="y14m08">
                <span>
                    <a href="http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX"> SARIMAX</a>
                </span>
            </td>
            <td class="y15m04">
                <span>
                    <a href="https://github.com/statsmodels/statsmodels/pull/2432"> Unobserved Components</a>
                </span>
            </td>
            <td class="y15m06">
                <span>
                    <a href="https://github.com/statsmodels/statsmodels/pull/2431"> Diagnostics</a>
                </span>
            </td>
            <td class="y15m07">
                <span>
                    <a href="https://github.com/statsmodels/statsmodels/pull/2563"> VAR, Dynamic Factors</a>
                </span>
            </td>
            <td class="y15m08">
                <span>
                    <a href="./topics/state_space_mh.html"> Metropolis-Hastings</a>
                </span>
            </td>
        </tr>
        <tr class="items">
            <td class="y14m05">
                <span class="spacer">  |</span>
                <span>May, 2014</span>
            </td>
            <td class="y14m08">
                <span class="spacer">  |</span>
                <span>August, 2014</span>
            </td>
            <td class="y15m04">
                <span class="spacer">  |</span>
                <span>April, 2015</span>
            </td>
            <td class="y15m06">
                <span class="spacer">  |</span>
                <span>June, 2015</span>
            </td>
            <td class="y15m07">
                <span class="spacer">  |</span>
                <span>July, 2015</span>
            </td>
            <td class="y15m08">
                <span class="spacer">  |</span>
                <span>August, 2015</span>
            </td>
        </tr>
    </tbody>
</table>
</div>

## Statsmodels: State space models and the Kalman filter

[View on Github](https://github.com/statsmodels/statsmodels/pull/2250){:target="_blank"}

**Summary**: I contributed a module to the Statsmodels project which
allows (1) specification of state space models, (2) fast
Kalman filtering of those models, and (3) easy estimation
of parameters via maximum likelihood estimation. See
below for details.

For a longer description of these types of models, a discussion of the
implementation in Statsmodels, and example code, see the following link (note:
the content at this link used to be on this page): **[Implementing state space models for Statsmodels](./topics/implementing_state_space.html)**

For more information about state space models in Python::

- **[State space models in Python](./topics/state_space_python.html)** - this post describes the general approach that was taken in creating the `statespace` approach in Statsmodels, and gives example code for the local linear trend model.
- **[State space diagnostics](./topics/state_space_diagnostics.html)** - this post describes diagnostic statistics for state space models.
- **[Bayesian state space estimation via Metropolis-Hastings](./topics/state_space_mh.html)** - this post describes estimation of state space models via Metropolis-Hastings (Bayesian posterior simulation).
- **[Estimating an RBC model](./topics/estimating_rbc.html)** - this post provides an example of setting up, solving, and estimating a simple RBC model using the `statespace` library in Statsmodels.

</section>

<section id="markov-switching">

## Statsmodels: Markov switching dynamic regression and autoregression

[View on Github](https://github.com/statsmodels/statsmodels/pull/2980){:target="_blank"}

**Summary**: I contributed `MarkovRegression` and `MarkovAutoregression`
classes to the Statsmodels project allowing maximum likelihood estimation of
these classes of models.

For more information about these Markov switching models:

- **[Markov dynamic regression models](./topics/markov_regression.html)** - Examples and discussion of Markov switching dynamic regression models.
- **[Markov autoregression models](./topics/markov_autoregression.html)** - Examples and discussion of Markov switching autoregression models.

</section>

<section id="wishart">

## Scipy: Wishart random variables and sampling

[View Scipy documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html#scipy.stats.wishart){:target="_blank"}

**Summary**: I contributed `wishart` and `invwishart` classes to the Scipy
project allowing evaluation of properties of these random variables (PDF,
entropy, etc.) as well as the drawing of random samples from these
distributions. These can be useful in a variety of settings, including Gibbs
sampling approach to estimating covariance matrices in state space models.

</section>
