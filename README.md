# Logistic Regression - Kaggle SpaceShip Titanic 

## Overview

We want to solve the Kaggle's [SpaceShip Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) challenge, in order to practice logistic regression for binary classification.


> Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.\
\
The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.\
\
While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before.\
Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!\
\
To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.\
Help save them and change history!\
\
-- <cite> Addison Howard, Ashley Chow, Ryan Holbrook

## Workflow

### Model

We will use a sigmoid function $g$ to compute the output of our model :

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

With $z$ being a linear equation :

$$
z = \vec{w}\cdot \vec{x} + b
$$

Resulting in model $f$ :

$$
f_{\vec{w}, b}(\vec{x}) = g(\vec{w}\cdot \vec{x} + b) = \frac {1} {1 + e ^ {-(\vec{w} \cdot \vec{x} + b)}}
$$

### Data processing

- Select relevant features to solve the problem
- Tokenize string features
- Scale the data to range between $[0\cdots 1]$
### Training

#### Loss function

The loss function used to measure the difference between the prediction for one input and its associated target :

$$
L(f_{\vec{w}, b}(\vec{x^{(i)}}), y^{(i)}) = \begin{cases}
    \begin{array}{ll}
        -\log(f_{\vec{w}, b}(\vec{x}^{(i)})) & \text{if  } y^{(i)}=1 \\
        -\log(1 - f_{\vec{w}, b}(\vec{x}^{(i)})) & \text{if  } y^{(i)}=0
    \end{array}
\end{cases}
$$

Simplified to :

$$
L(f_{\vec{w}, b}(\vec{x^{(i)}}), y^{(i)}) = -y^{(i)}\log(f_{\vec{w}, b}(\vec{x}^{(i)})) - (1 - y^{(i)})\log(1-f_{\vec{w}, b}(\vec{x}^{(i)}))
$$

#### Regularization function

We will use a regularization function to minimize the overfitting of the data during training:

$$
R(\vec{w}) = \frac {\lambda} {2m} \sum^n_{j=1}w^2_j
$$

With $\lambda$ being the regularization parameter and $m$ the number of examples in the dataset.

#### Cost function

We measure the overall error of the model for the whole training set:

$$
J(\vec{w}, b) = \frac{1}{m}\sum^m_{i=1}[L(f_{\vec{w}, b}(\vec{x}), y^{(i)})] + R(\vec{w})
$$

#### Cost derivatives

Partial derivative of $J$ with respect to $\vec{w}_j$ :

$$
\frac {\delta} {\delta \vec{w}_j} J(\vec{w}, b) = \frac {1} {m} \sum^m_{i=1}[(f_{\vec{w}, b}(\vec{x}) - y^{(i)})x^{(i)}_j] + \frac {\lambda} {m}w_j
$$

Partial derivative of $J$ with respect to $\vec{w}_j$ :

$$
\frac {\delta} {\delta b} J(\vec{w}, b) = \frac {1} {m} \sum^m_{i=1}(f_{\vec{w}, b}(\vec{x}) - y^{(i)})
$$

#### Descent

We will repeat until convergence is met or maximum number of iteration is reached, with $\alpha$ being the model learning rate :

$$
\begin{aligned}
    & \vec{w}_j = \vec{w}_j - \alpha \frac{\delta}{\delta \vec{w}_j} J(\vec{w}, b)\\
    & b=b - \alpha \frac{\delta}{\delta b} J(\vec{w}, b)
\end{aligned}\\
$$
