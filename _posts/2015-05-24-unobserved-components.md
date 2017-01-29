---
layout: post
title:  "Unobserved components"
date:   2015-05-24 09:57:01 -07:00
categories: time-series unobserved-components
permalink: /topics/unobserved_components.html
redirect_from: /posts/unobserved_components.html
---

The first model considered in the state space models GSoC 2015 project is the class of univariate unobserved components models. This blog post lays out the general structure and the different variations that will be allowed.

The basic unobserved components (or structural time series) model can be written (see Durbin and Koopman 2012, Chapter 3 for notation and additional details):

$$
y_t = \underbrace{\mu_{t}}_{\text{trend}} + \underbrace{\gamma_{t}}_{\text{seasonal}} + \underbrace{c_{t}}_{\text{cycle}} + \underbrace{\varepsilon_t}_{\text{irregular}}
$$

where different specifications for the different individual components can support a range of models.

<hr />

### Trend

The trend component is a dynamic extension of a regression model that includes an intercept and linear time-trend.

$$
\begin{align}
\underbrace{\mu_{t+1}}_{\text{level}} & = \mu_t + \nu_t + \eta_{t+1} \qquad & \eta_{t+1} \sim N(0, \sigma_\eta^2) \\\\
\underbrace{\nu_{t+1}}_{\text{trend}} & = \nu_t + \zeta_{t+1} & \zeta_{t+1} \sim N(0, \sigma_\zeta^2) \\
\end{align}
$$

where the level is a generalization of the intercept term that can dynamically vary across time, and the trend is a generalization of the time-trend such that the slope can dynamically vary across time.

For both elements (level and trend), we can consider models in which:

- The element is included vs excluded (if the trend is included, there must also be a level included).
- The element is deterministic vs stochastic (i.e. whether or not the variance on the error term is confined to be zero or not)

The only additional parameters to be estimated via MLE are the variances of any included stochastic components.

This leads to the following specifications:

|                                                                      | Level | Trend | Stochastic Level | Stochastic Trend |
|----------------------------------------------------------------------|-------|-------|------------------|------------------|
| Constant                                                             | ✓     |       |                  |                  |
| Local Level <br /> (random walk)                                     | ✓     |       | ✓                |                  |
| Deterministic trend                                                  | ✓     | ✓     |                  |                  |
| Local level with deterministic trend <br /> (random walk with drift) | ✓     | ✓     | ✓                |                  |
| Local linear trend                                                   | ✓     | ✓     | ✓                | ✓                |
| Smooth trend <br /> (integrated random walk)                         | ✓     | ✓     |                  | ✓                |

<hr />

### Seasonal

The seasonal component is written as:

$$
\gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \qquad \omega_t \sim N(0, \sigma_\omega^2)
$$

The periodicity (number of seasons) is `s`, and the defining character is that (without the error term), the seasonal components sum to zero across one complete cycle. The inclusion of an error term allows the seasonal effects to vary over time.

The variants of this model are:

- The periodicity `s`
- Whether or not to make the seasonal effects stochastic.

If the seasonal effect is stochastic, then there is one additional parameter to estimate via MLE (the variance of the error term).

<hr />

### Cycle

The cyclical component is intended to capture cyclical effects at time frames much longer than captured by the seasonal component. For example, in economics the cyclical term is often intended to capture the business cycle, and is then expected to have a period between "1.5 and 12 years" (see Durbin and Koopman).

The cycle is written as:

$$
\begin{align}
c_{t+1} & = c_t \cos \lambda_c + c_t^* \sin \lambda_c + \tilde \omega_t \qquad & \tilde \omega_t \sim N(0, \sigma_{\tilde \omega}^2) \\\\
c_{t+1}^* & = -c_t \sin \lambda_c + c_t^* \cos \lambda_c + \tilde \omega_t^* & \tilde \omega_t^* \sim N(0, \sigma_{\tilde \omega}^2)
\end{align}
$$

The parameter `lambda_c` (the frequency of the cycle) is an additional parameter to be estimated by MLE. If the seasonal effect is stochastic, then there is one another parameter to estimate (the variance of the error term - note that both of the error terms here share the same variance, but are assumed to have independent draws).

<hr />

### Irregular

The irregular component is assumed to be a white noise error term. Its variance is a parameter to be estimated by MLE.

<hr />

## Extensions

The following extensions may be included in the model depending on time-constraints:

<hr />

### Autoregressive irregular

In some cases, we may want to generalize the irregular component to allow for autoregressive effects:

$$
\varepsilon_t = \rho(L) \varepsilon_{t-1} + \epsilon_t
$$

In this case, the autoregressive parameters would be estimated via MLE.

<hr />

### Regression effects

We may want to allow for explanatory variables by including additional terms

$$
\sum_{j=1}^k \beta_j x_{jt}
$$

(where a futher extension would allow time-varying regression parameters) or for intervention effects by including

$$
\begin{align}
\delta w_t \qquad \text{where} \qquad w_t & = 0, \qquad t < \tau, \\\\
& = 1, \qquad t \ge \tau
\end{align}
$$

These additional parameters could be estimated via MLE or by including them as components of the state space formulation.

