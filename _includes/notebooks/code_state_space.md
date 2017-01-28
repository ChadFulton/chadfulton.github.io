
An astonishing variety of time series econometrics problems can
be handled in one way or another by putting a model into state
space form and applying the Kalman filter, providing optimal
estimates of latent state variables conditioning on observed
data and the loglikelihood of parameters. Better still, writing
code to run through the Kalman filter recursions is very
straightforward in many of the popular software packages (e.g.
Python, MATLAB) and can be accomplished in fewer than 50 lines of code.

Considering a time-invariant state-space model such
as<sup>3</sup>:

$$
\begin{align}
y_t & = Z \alpha_t + \varepsilon_t \qquad & \varepsilon_t \sim N(0, H) \\\
\alpha_{t+1} & = T \alpha_t + \eta_t \qquad & \eta_t \sim N(0, Q) \\\
\alpha_0 & \sim N(a_0, P_0) ~ \text{known}
\end{align}
$$

the Kalman filter can be written as


```python
import numpy as np

def kalman_filter(y, Z, H, T, Q, a_0, P_0):
    # Dimensions
    k_endog, nobs = y.shape
    k_states = T.shape[0]

    # Allocate memory for variables
    filtered_state = np.zeros((k_states, nobs))
    filtered_state_cov = np.zeros((k_states, k_states, nobs))
    predicted_state = np.zeros((k_states, nobs+1))
    predicted_state_cov = np.zeros((k_states, k_states, nobs+1))
    forecast = np.zeros((k_endog, nobs))
    forecast_error = np.zeros((k_endog, nobs))
    forecast_error_cov = np.zeros((k_endog, k_endog, nobs))
    loglikelihood = np.zeros((nobs+1,))

    # Copy initial values to predicted
    predicted_state[:, 0] = a_0
    predicted_state_cov[:, :, 0] = P_0

    # Kalman filter iterations
    for t in range(nobs):

        # Forecast for time t
        forecast[:, t] = np.dot(Z, predicted_state[:, t])

        # Forecast error for time t
        forecast_error[:, t] = y[:, t] - forecast[:, t]

        # Forecast error covariance matrix and inverse for time t
        tmp1 = np.dot(predicted_state_cov[:, :, t], Z.T)
        forecast_error_cov[:, :, t] = (
            np.dot(Z, tmp1) + H
        )
        forecast_error_cov_inv = np.linalg.inv(forecast_error_cov[:, :, t])
        determinant = np.linalg.det(forecast_error_cov[:, :, t])

        # Filtered state for time t
        tmp2 = np.dot(forecast_error_cov_inv, forecast_error[:,t])
        filtered_state[:, t] = (
            predicted_state[:, t] +
            np.dot(tmp1, tmp2)
        )

        # Filtered state covariance for time t
        tmp3 = np.dot(forecast_error_cov_inv, Z)
        filtered_state_cov[:, :, t] = (
            predicted_state_cov[:, :, t] -
            np.dot(
                np.dot(tmp1, tmp3),
                predicted_state_cov[:, :, t]
            )
        )

        # Loglikelihood
        loglikelihood[t] = -0.5 * (
            np.log((2*np.pi)**k_endog * determinant) +
            np.dot(forecast_error[:, t], tmp2)
        )

        # Predicted state for time t+1
        predicted_state[:, t+1] = np.dot(T, filtered_state[:, t])

        # Predicted state covariance matrix for time t+1
        tmp4 = np.dot(T, filtered_state_cov[:, :, t])
        predicted_state_cov[:, :, t+1] = np.dot(tmp4, T.T) + Q
        
        predicted_state_cov[:, :, t+1] = (
            predicted_state_cov[:, :, t+1] + predicted_state_cov[:, :, t+1].T
        ) / 2

    return (
        filtered_state, filtered_state_cov,
        predicted_state, predicted_state_cov,
        forecast, forecast_error, forecast_error_cov,
        loglikelihood
    )
```

So why then did I write nearly 15,000 lines of code to
contribute Kalman filtering and state-space models to the
Statsmodels project?

1. **Performance**: It should run fast
2. **Wrapping**: It should be easy to use
3. **Testing**: It should run correctly

### Performance

The Kalman filter basically consists of iterations (loops) and
matrix operations. It is well known that loops perform poorly
in interpreted languages like Python<sup>1</sup>, and also that
matrix operations are ultimately performed by the highly
optimized [BLAS](http://www.netlib.org/blas/) and
[LAPACK](http://www.netlib.org/lapack/) libraries, regardless
of the high-level programming language used.<sup>2</sup> This
suggests two things:

- Fast code should be compiled (not interpreted)
- Fast code should call the BLAS / LAPACK libraries as soon
   as possible (not through intermediate functions)

These two things are possible using
[Cython](http://cython.org/), a simple extension of Python
syntax that allows compilation to C and direct interaction with
BLAS and LAPACK. All of the heavy lifting of the Kalman
filtering I contributed to Statsmodels is performed in Cython,
which allows for very fast execution.

It might seem like this approach eliminates the whole benefit
of using a high-level language like Python - in fact, why not
just use C or Fortran if we're going to ultimately compile the
code? First, Cython is quite similar to Python, so future
maintenance is easier, but more importantly end-user Python
code can interact with it directly. In this way, we get the
best of both worlds: the speed of compiled code where
performance is needed and the ease of interpreted code where it
isn't.

An $AR(1)$ model can be written in state space form as

$$
\begin{align}
    y_t & = \alpha_t \\\
    \alpha_{t+1} & = \phi_1 \alpha_t + \eta_t \qquad \eta_t \sim N(0, \sigma_\eta^2)
\end{align}
$$

and it can specified in Python and the Kalman filter applied
using the following code:



```python
from scipy.signal import lfilter

# Parameters
nobs = 100
phi = 0.5
sigma2 = 1.0

# Example dataset
np.random.seed(1234)
eps = np.random.normal(scale=sigma2**0.5, size=nobs)
y = lfilter([1], [1, -phi], eps)[np.newaxis, :]

# State space
Z = np.array([1.])
H = np.array([0.])
T = np.array([phi])
Q = np.array([sigma2])

# Initial state distribution
a_0 = np.array([0.])
P_0 = np.array([sigma2 / (1 - phi**2)])

# Run the Kalman filter
res = kalman_filter(y, Z, H, T, Q, a_0, P_0)
```

Comparing the above Kalman filter with the implementation in
Statsmodels for the $AR(1)$ model yields the following runtimes
in milliseconds for a single filter application, where $nobs$
is the length of the time series (reasonable measures were
taken to ensure these timings are meaningful, but not
extraordinary measures):


| `nobs`        | Python (ms) | MATLAB (ms) | Cython (ms)   |
|---------------|-------------|-------------|---------------|
| $10$   &nbsp; | $0.742$     | $0.326$     | $0.106$       |
| $10^2$        | $6.39$      | $3.040$     | $0.161$       |
| $10^3$        | $67.1$      | $32.5$      | $0.668$       |
| $10^4$        | $662.0$     | $311.3$     | $6.1$         |

Across hundreds or thousands of iterations (as in maximum
likelihood estimation or MCMC methods), these differences can
be substantial. Also, other Kalman filtering methods, such as
the univariate approach of Koopman and Durbin (2000) used with
large dimensional observations $y_t$, can add additional inner
loops, increasing the importance of compiled code.

### Wrapping

One of the main reaons that using Python or MATLAB is
preferrable to C or Fortran is that code in higher-level 
lanaguages is more expressive and more readable. Even though
the performance sensitive code has been written in Cython, we
want to take advantage of the high-level language features in
Python proper to make specifying, filtering, and estimating
parameters of state space models as natural as possible. For
example, an $ARMA(1,1)$ model can be written in state-space
form as

$$
\begin{align}
    y_t & = \begin{bmatrix} 1 & \theta_1 \end{bmatrix} \begin{bmatrix} \alpha_{1t} \\ \alpha_{2t} \end{bmatrix} \\\
    \begin{bmatrix} \alpha_{1t+1} \\ \alpha_{2t+1} \end{bmatrix} & = \begin{bmatrix}
    \phi_1 & 0 \\
    1 & 0
    \end{bmatrix} \begin{bmatrix} \alpha_{1t} \\ \alpha_{2t} \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t \qquad \eta_t \sim N(0, \sigma_\eta^2)
\end{align}
$$

where $\theta\_1$, $\phi\_1$, and $\sigma\_\eta^2$ are unknown
parameters. Estimating them via MLE has been made very
easy in the Statsmodels state space library; the model can be
specified and estimated with the following code

**Note**: this code has been updated on July 31, 2015 to
reflect an update to the Statsmodels code base.

**Note**: this code has been updated on June 17, 2016 to
reflect a further update to the Statsmodels code base, and
also to estimate an ARMA(1,1) model as shown above.


```python
import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm

# True model parameters for an AR(1)
nobs = int(1e3)
true_phi = 0.5
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model for an ARMA(1,1)
class ARMA11(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(ARMA11, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1., 0]
        self['transition'] = [[0, 0],
                              [1., 0]]
        self['selection', 0, 0] = 1.

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(ARMA11, self).update(params, transformed, **kwargs)

        self['design', 0, 1] = params[0]
        self['transition', 0, 0] = params[1]
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.,0.,1]  # these are very simple

# Create and fit the model
mod = ARMA11(endog)
res = mod.fit()
print(res.summary())
```

                               Statespace Model Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1000
    Model:                         ARMA11   Log Likelihood               -1389.992
    Date:                Sun, 22 Jan 2017   AIC                           2785.984
    Time:                        15:40:18   BIC                           2800.707
    Sample:                             0   HQIC                          2791.580
                                   - 1000                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    param.0       -0.0203      0.072     -0.284      0.776      -0.161       0.120
    param.1        0.4617      0.065      7.140      0.000       0.335       0.588
    param.2        0.9436      0.042     22.413      0.000       0.861       1.026
    ===================================================================================
    Ljung-Box (Q):                       25.04   Jarque-Bera (JB):                 0.16
    Prob(Q):                              0.97   Prob(JB):                         0.92
    Heteroskedasticity (H):               1.05   Skew:                            -0.03
    Prob(H) (two-sided):                  0.63   Kurtosis:                         3.01
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Whereas the above example showed an ad-hoc creation and
estimation of a specific model, the power of object-oriented
programming in Python can be leveraged to create generic and
reusable estimation classes. For example, for the common class
of (Seasonal) Autoregressive Integrated Moving Average models
(optionally with exogenous regressors), an `SARIMAX` class has
been written to automate the creation and estimation of those
types of models. For example, an
$SARIMA(1,1,1) \times (0,1,1,4)$ model of GDP can be specified
and estimated as (an added bonus is that we can download the
GDP data on-the-fly from FRED using Pandas):

**Note**: this code has been updated on June 17, 2016 to
reflect an update to the Statsmodels code base and to use the
`pandas_datareader` package.


```python
import statsmodels.api as sm
from pandas_datareader.data import DataReader

gdp = DataReader('GDPC1', 'fred', start='1959', end='12-31-2014')

# Create the model, here an SARIMA(1,1,1) x (0,1,1,4) model
mod = sm.tsa.SARIMAX(gdp, order=(1,1,1), seasonal_order=(0,1,1,4))

# Fit the model via maximum likelihood
res = mod.fit()
print(res.summary())
```

                                     Statespace Model Results                                
    =========================================================================================
    Dep. Variable:                             GDPC1   No. Observations:                  224
    Model:             SARIMAX(1, 1, 1)x(0, 1, 1, 4)   Log Likelihood               -1222.487
    Date:                           Sun, 22 Jan 2017   AIC                           2452.974
    Time:                                   15:40:39   BIC                           2466.620
    Sample:                               01-01-1959   HQIC                          2458.482
                                        - 10-01-2014                                         
    Covariance Type:                             opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.6636      0.117      5.695      0.000       0.435       0.892
    ma.L1         -0.3247      0.143     -2.263      0.024      -0.606      -0.043
    ma.S.L4       -0.9332      0.034    -27.594      0.000      -0.999      -0.867
    sigma2      3981.5479    286.690     13.888      0.000    3419.645    4543.450
    ===================================================================================
    Ljung-Box (Q):                       44.38   Jarque-Bera (JB):               140.86
    Prob(Q):                              0.29   Prob(JB):                         0.00
    Heteroskedasticity (H):               2.94   Skew:                            -0.81
    Prob(H) (two-sided):                  0.00   Kurtosis:                         6.58
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


This type of built-in model should be familiar to those who
work with programs like Stata (which also has a built-in
SARIMAX model). The benefit of Python and Statsmodels is that
you can build *your own* classes of models which behave just
as smoothly and seamlessly as those that are "built-in". By
building on top of the state space functionality in
Statsmodels, you get a lot for free while still retaining the
flexibility to write any kind of model you want.

For example, a local linear trend model can be created for
re-use in the following way:

**Note**: this code has been updated on June 17, 2016 to
reflect an update to the Statsmodels code base.


```python
"""
Univariate Local Linear Trend Model
"""
import pandas as pd

class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, trend=True):
        # Model properties
        self.trend = trend

        # Model order
        k_states = 2
        k_posdef = 1 + self.trend

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
        )

        # Initialize the matrices
        self['design'] = np.array([1, 0])
        self['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self['selection'] = np.eye(k_states)[:,:k_posdef]

        # Initialize the state space model as approximately diffuse
        self.initialize_approximate_diffuse()
        # Because of the diffuse initialization, burn first two loglikelihoods
        self.loglikelihood_burn = 2

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

        # The parameters depend on whether or not we have a trend
        param_names = ['sigma2.measurement', 'sigma2.level']
        if self.trend:
            param_names += ['sigma2.trend']
        self._param_names = param_names

    @property
    def start_params(self):
        return [0.1] * (2 + self.trend)

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self['obs_cov',0,0] = params[0]

        # State covariance
        self[self._state_cov_idx] = params[1:]
```

Now, we have a generic class that can fit local linear trend
models (if `trend=True`) and also local level models (if
`trend=False`). For example, we can model the annual flow
volume of the Nile river using a local linear trend model:


```python
y = sm.datasets.nile.load_pandas().data
y.index = pd.date_range('1871', '1970', freq='AS')

mod1 = LocalLinearTrend(y['volume'], trend=True)
res1 = mod1.fit()
print res1.summary()
```

                               Statespace Model Results                           
    ==============================================================================
    Dep. Variable:                 volume   No. Observations:                  100
    Model:               LocalLinearTrend   Log Likelihood                -629.858
    Date:                Sun, 22 Jan 2017   AIC                           1265.716
    Time:                        15:41:27   BIC                           1273.532
    Sample:                    01-01-1871   HQIC                          1268.879
                             - 01-01-1970                                         
    Covariance Type:                  opg                                         
    ======================================================================================
                             coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    sigma2.measurement  1.469e+04   2756.914      5.330      0.000    9291.260    2.01e+04
    sigma2.level        1747.4389   1211.919      1.442      0.149    -627.879    4122.756
    sigma2.trend        3.097e-06      4.254   7.28e-07      1.000      -8.339       8.339
    ===================================================================================
    Ljung-Box (Q):                       36.16   Jarque-Bera (JB):                 0.05
    Prob(Q):                              0.64   Prob(JB):                         0.98
    Heteroskedasticity (H):               0.62   Skew:                             0.05
    Prob(H) (two-sided):                  0.17   Kurtosis:                         3.05
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


It looks as though the presense of a stochastic trend
is not adding anything to the model (and the parameter is not
estimated well in any case) - refitting without the trend is
easy:


```python
mod2 = LocalLinearTrend(y['volume'], trend=False)
res2 = mod2.fit()
print res2.summary()
```

                               Statespace Model Results                           
    ==============================================================================
    Dep. Variable:                 volume   No. Observations:                  100
    Model:               LocalLinearTrend   Log Likelihood                -629.858
    Date:                Sun, 22 Jan 2017   AIC                           1263.717
    Time:                        15:41:45   BIC                           1268.927
    Sample:                    01-01-1871   HQIC                          1265.825
                             - 01-01-1970                                         
    Covariance Type:                  opg                                         
    ======================================================================================
                             coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    sigma2.measurement  1.472e+04   2734.512      5.383      0.000    9360.283    2.01e+04
    sigma2.level        1742.4785   1117.075      1.560      0.119    -446.949    3931.906
    ===================================================================================
    Ljung-Box (Q):                       36.17   Jarque-Bera (JB):                 0.04
    Prob(Q):                              0.64   Prob(JB):                         0.98
    Heteroskedasticity (H):               0.62   Skew:                             0.04
    Prob(H) (two-sided):                  0.17   Kurtosis:                         3.05
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Instead of constructing our own custom class, this particular example could be estimated using the `UnobservedComponents` model in the Statsmodels state space library.

### Testing

It is no good to have fast code that is easy to use if it gives
the wrong answer. For that reason, a large part of creating
production ready code is constructing unit tests comparing
the module's output to known values to make sure everything
works. The state space model code in Statsmodels has 455 unit
tests covering everything from the filter output
(`filtered_state`, `logliklelihood`, etc.) to state space
creation (e.g. the `SARIMAX` class) and maximum likelihood
estimation (estimated parameters, maximized likelihood values,
standard errors, etc.).

### Bibliography

Durbin, James, and Siem Jan Koopman. 2012.
Time Series Analysis by State Space Methods: Second Edition.
Oxford University Press.

Koopman, S. J., and J. Durbin. 2000.
“Fast Filtering and Smoothing for Multivariate State Space Models.”
Journal of Time Series Analysis 21 (3): 281–96.

### Footnotes

[1] This can be improved with a JIT compiler like
[Numba](http://numba.pydata.org/).

[2] Python, MATLAB, Mathematica, Stata, Gauss, Ox, etc. all
ultimately rely on BLAS and LAPACK libraries for performing
operations on matrices.

[3] See Durbin and Koopman (2012) for notation.

[4] A [proposal](http://legacy.python.org/dev/peps/pep-0465/)
is in place to create an infix matrix multiplication
operator in Python

