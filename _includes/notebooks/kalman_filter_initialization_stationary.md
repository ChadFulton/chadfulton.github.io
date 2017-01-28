
### The initialization problem

The Kalman filter is a recursion for optimally making inferences about an unknown state variable given a related observed variable. In particular, if the state variable at time $t$ is represented by $\alpha_t$, then the (linear, Gaussian) Kalman filter takes as input the mean and variance of that state conditional on observations up to time $t-1$ and provides as output the filtered mean and variance of the state at time $t$ and the predicted mean and variance of the state at time $t$.

More concretely, we denote (see Durbin and Koopman (2012) for all notation)

$$
\begin{align}
\alpha_t \mid Y_{t-1} & \sim N(a_t, P_t) \\
\alpha_t \mid Y_{t} & \sim N(a_{t|t}, P_{t|t}) \\
\alpha_{t+1} \mid Y_{t} & \sim N(a_{t+1}, P_{t+1}) \\
\end{align}
$$

Then the inputs to the Kalman filter recursion are $a_t$ and $P_t$ and the outputs are $a_{t \mid t}, P_{t \mid t}$ (called *filtered* values) and $a_{t+1}, P_{t+1}$ (called *predicted* values).

This process is done for $t = 1, \dots, n$. While the predicted values as outputs of the recursion are available as inputs to subsequent iterations, an important question is *initialization*: what values should be used as inputs to start the very first recursion.

Specifically, when running the recursion for $t = 1$, we need as inputs $a_1, P_1$. These values define, respectively, the expectation and variance / covariance matrix for the initial state $\alpha_1 \mid Y_0$. Here, though, $Y_0$ denotes the observation of *no data*, so in fact we are looking for the *unconditional* expectation and variance / covariance matrix of $\alpha_1$. The question is how to find these.

In general this is a rather difficult problem (for example for non-stationary proceses) but for stationary processes, an analytic solution can be found.

### Stationary processes

A (covariance) stationary process is, very roughly speaking, one for which the mean and covariances are not time-dependent. What this means is that we can solve for the unconditional expectation and variance explicity (this section results from Hamilton (1994), Chapter 13)

The state equation for a state-space process (to which the Kalman filter is applied) is

$$
\alpha_{t+1} = T \alpha_t + \eta_t
$$

Below I set up the elements of a typical state equation like that which would be found in the ARMA case, where the transition matrix $T$ is a sort-of companion matrix. I'm setting it up in such a way that I'll be able to adjust the dimension of the state, so we can see how some of the below methods scale.


```python
import numpy as np
from scipy import linalg

def state(m=10):
    T = np.zeros((m, m), dtype=complex)
    T[0,0] = 0.6 + 1j
    idx = np.diag_indices(m-1)
    T[(idx[0]+1, idx[1])] = 1
    
    Q = np.eye(m)
    
    return T, Q
```

#### Unconditional mean

Taking unconditional expectations of both sides yields:

$$
E[\alpha_{t+1}] = T E[ \alpha_t] + E[\eta_t]
$$

or $(I - T) E[\alpha_t] = 0$ and given stationarity this means that the unique solution is $E[\alpha_t] = 0$ for all $t$. Thus in initializing the Kalman filter, we set $a_t = E[\alpha_t] = 0$.

#### Unconditional variance / covariance matrix

Slightly more tricky is the variance / covariance matrix. To find it (as in Hamilton) post-multiply by the transpose of the state and take expectations:

$$
E[\alpha_{t+1} \alpha_{t+1}'] = E[(T \alpha_t + \eta_t)(\alpha_t' T' + \eta_t')]
$$

This yields an equation of the form (denoting by $\Sigma$ and $Q$ the variance / covariance matrices of the state and disturbance):

$$
\Sigma = T \Sigma T' + Q
$$

Hamilton then shows that this equation can be solved as:

$$
vec(\Sigma) = [I - (T \otimes T)]^{-1} vec(Q)
$$

where $\otimes$ refers to the Kronecker product. There are two things that jump out about this equation:

1. It can be easily solved. In Python, it would look something like:
   ```python
   m = T.shape[0]
   Sigma = np.linalg.inv(np.eye(m**2) - np.kron(T, T)).dot(Q.reshape(Q.size, 1)).reshape(n,n)
   ```
2. It will scale very poorly (in terms of computational time) with the dimension of the state-space ($m$). In particular, you have to take the inverse of an $m^2 \times m^2$ matrix.

Below I take a look at the timing for solving it this way using the code above (`direct_inverse`) and using built-in scipy direct method (which uses a linear solver rather than taking the inverse, so it is a bit faster)s


```python
def direct_inverse(A, Q):
    n = A.shape[0]
    return np.linalg.inv(np.eye(n**2) - np.kron(A,A.conj())).dot(Q.reshape(Q.size, 1)).reshape(n,n)

def direct_solver(A, Q):
    return linalg.solve_discrete_lyapunov(A, Q)

# Example
from numpy.testing import assert_allclose
np.set_printoptions(precision=10)
T, Q = state(3)
sol1 = direct_inverse(T, Q)
sol2 = direct_solver(T, Q)

assert_allclose(sol1,sol2)
```


```python
# Timings for m=1
T, Q = state(1)
%timeit direct_inverse(T, Q)
%timeit direct_solver(T, Q)
```

    The slowest run took 4.63 times longer than the fastest. This could mean that an intermediate result is being cached 
    10000 loops, best of 3: 50.9 µs per loop
    The slowest run took 168.35 times longer than the fastest. This could mean that an intermediate result is being cached 
    10000 loops, best of 3: 74.3 µs per loop



```python
# Timings for m=5
T, Q = state(5)
%timeit direct_inverse(T, Q)
%timeit direct_solver(T, Q)
```

    10000 loops, best of 3: 138 µs per loop
    10000 loops, best of 3: 136 µs per loop



```python
# Timings for m=10
T, Q = state(10)
%timeit direct_inverse(T, Q)
%timeit direct_solver(T, Q)
```

    1000 loops, best of 3: 1.75 ms per loop
    1000 loops, best of 3: 285 µs per loop



```python
# Timings for m=50
T, Q = state(50)
%timeit direct_inverse(T, Q)
%timeit direct_solver(T, Q)
```

    1 loops, best of 3: 12.5 s per loop
    100 loops, best of 3: 5.07 ms per loop


### Lyapunov equations

As you can notice by looking at the name of the scipy function, the equation describing the unconditional variance / covariance matrix, $\Sigma = T \Sigma T' + Q$ is an example of a discrete Lyapunov equation.

One place to turn to improve performance on matrix-related operations is to the underlying Fortran linear algebra libraries: BLAS and LAPACK; if there exists a special-case solver for discrete time Lyapunov equations, we can call that function and be done.

Unfortunately, no such function exists, but what does exist is a special-case solver for *Sylvester* equations (\*trsyl), which are equations of the form $AX + XB = C$. Furthermore, the *continuous* Lyapunov equation, $AX + AX^H + Q = 0$ is a special case of a Sylvester equation. Thus if we can transform the discrete to a continuous Lyapunov equation, we can then solve it quickly as a Sylvester equation.

The current implementation of the scipy discrete Lyapunov solver does not do that, although their continuous solver `solve_lyapunov` does call `solve_sylvester` which calls \*trsyl. So, we need to find a transformation from discrete to continuous and directly call `solve_lyapunov` which will do the heavy lifting for us.

It turns out that there are several transformations that will do it. See Gajic, Z., and M.T.J. Qureshi. 2008. for details. Below I present two bilinear transformations, and show their timings.


```python
def bilinear1(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    B = np.linalg.inv(A - eye).dot(A + eye)
    res = linalg.solve_lyapunov(B.conj().T, -Q)
    return 0.5*(B - eye).conj().T.dot(res).dot(B - eye)

def bilinear2(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    AI_inv = np.linalg.inv(A + eye)
    B = (A - eye).dot(AI_inv)
    C = 2*np.linalg.inv(A.conj().T + eye).dot(Q).dot(AI_inv)
    return linalg.solve_lyapunov(B.conj().T, -C)

# Example:
T, Q = state(3)
sol3 = bilinear1(T, Q)
sol4 = bilinear2(T, Q)

assert_allclose(sol1,sol3)
assert_allclose(sol3,sol4)
```


```python
# Timings for m=1
T, Q = state(1)
%timeit bilinear1(T, Q)
%timeit bilinear2(T, Q)
```

    10000 loops, best of 3: 182 µs per loop
    10000 loops, best of 3: 193 µs per loop



```python
# Timings for m=5
T, Q = state(5)
%timeit bilinear1(T, Q)
%timeit bilinear2(T, Q)
```

    10000 loops, best of 3: 199 µs per loop
    1000 loops, best of 3: 216 µs per loop



```python
# Timings for m=10
T, Q = state(10)
%timeit bilinear1(T, Q)
%timeit bilinear2(T, Q)
```

    1000 loops, best of 3: 240 µs per loop
    1000 loops, best of 3: 271 µs per loop



```python
# Timings for m=50
T, Q = state(50)
%timeit bilinear1(T, Q)
%timeit bilinear2(T, Q)
```

    100 loops, best of 3: 2.36 ms per loop
    100 loops, best of 3: 2.78 ms per loop


Notice that this method does so well we can even try $m=500$.


```python
# Timings for m=500
T, Q = state(500)
%timeit bilinear1(T, Q)
%timeit bilinear2(T, Q)
```

    1 loops, best of 3: 1.55 s per loop
    1 loops, best of 3: 1.66 s per loop


### Final thoughts

The first thing to notice is *how much better* the bilinear transformations do as $m$ grows large. They are able to take advantage of the special formulation of the problem so as to avoid many calculations that a generic inverse (or linear solver) would have to do. Second, though, for small $m$, the original analytic solutions are actually better.

I have submitted a [pull request to Scipy](https://github.com/scipy/scipy/pull/3748) to augment the `solve_discrete_lyapunov` for large $m$ ($m >= 10$) using the second bilinear transformation to solve it as a Sylvester equation.
