# Gradient Descent
### Implementation of the Gradien Descent Optimizer for a 3-D Linear Regression problem.

The code above is for the challenge of [this](https://www.youtube.com/watch?v=xRJCOz3AfYY&t=9s) video by [Siraj Raval](https://github.com/llSourcell) on YouTube.

### Dependencies

* Numpy
* Pandas
* Matplotlib

### Dataset

The dataset used for this challenge has been downloaded form Kaggle, you can find it [here](https://www.kaggle.com/shivam2503/diamonds).
It's a CSV file with data about **Diamonds** such as: weight, color, width, height, depth, quality of the cut, price, clarity...

## Step 1: Data Exploration

As we can see in the `pd.read_csv(PATH)` function, the dataset has many samples and features. We can see that some of them are integers, some are floats and some are strings, such as the **cut**: Fair, Good, Very Good, Premium and Ideal.

<img src="imgs/dataframe.PNG">

## Step 2: Data Preparation

Since we're going to use a Linear Regression model and it's going to be optimized with a Gradient Descent, we're going to select 2-3 features. We're going to avoid string features since our model will work with numbers and we would have to convert them.

For the purpose of this challenge, we're going to select the price, the width and the mass of the diamonds since all of them are numerical features and *representative* (diamonds can be distinguished by them and they should be correlated).

Due to the huge amount of data (53000 samples), a randomized set of 500 samples is selected to perform the Linear Regression.

## Step 3: Analysis

We're going to build a Linear Regression Model. A Linear Regression model is a **linear model** with the following structure structure: 

<b>f(x) = w0 + &Sigma;<sub>j=1</sub><sup>d</sup> wj * xj</b>

We're going to find the weights of the linear equation that minimize the **Risk function**:

<b>(1/2N) * &Sigma;<sub>i=1</sub><sup>n</sup> (Yi - f(Xi))<sup>2</sup></b>

To do so, we'll need an optimization method, in this case, Gradient Descent. The Gradient Descent Algorithm consists in taking small steps towards the local minima, the point where our **Risk function has the lowest value**. To do so, we're going to compute the **gradients** (the partial derivative of each weight) and multiply them by the **learning rate** that we will define in order to update our weights.

We will repeat this process until:
1. The number of maximum steps we've defined is reached
2. The updates on the weights are so small that the model doesn't learn anymore.


## Step 4: Data Visualization

After performing a Linear Regression with a Gradient Descent as optimizer (starting all weights to 0, with a learning rate of 0.005 and 2500 iterations) we get the regression **hyperplane**, which is given by this equation: 

**Z = w0 + w1·X + w2·Y**

In order to make it easy to understand the so-called regression hyperplane, a data visualization is provided:

<img src="imgs/dataviz.PNG">

## Step 5: Conlusions

As we can see in the image above, both the mass and the width of the Diamonds contribute to their price. However, as we can see, the mass of the Diamonds is much more important when it comes to their price than their width.

We can appreciate something la a curve in the Graph. The higher the mass, the higher the width, and it makes sense since we're talking about a material with a fixed and unvariable density.

#### Special thanks to Siraj Raval, without who I would have never introduced myself in this awesome world.
