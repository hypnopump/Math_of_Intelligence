# Gradient Descent

## Implementation of the Gradien Descent Optimizer for a 3-D Liner Regression problem.

The code above is for the challenge of [this](https://www.youtube.com/watch?v=xRJCOz3AfYY&t=9s) video by [Siraj Raval](https://github.com/llSourcell) on YouTube.

### Dataset

The dataset used for this challenge has been downloaded form [Kaggle](https://www.kaggle.com). You can find it in [this link](https://www.kaggle.com/shivam2503/diamonds).

It's a CSV file with data about **Diamonds** such as:

* Weight
* Color
* Width
* Height
* Depth
* Quality ot the cut
* Price
* Clarity
* ...

### Data Visualization

After performing a Linear Regression with a Gradient Descent as optimizer (starting all weights to 0, with a learning rate of 0.005 and 2500 iterations) we get the regression **hyperplane**, which is given by this equation: 

**Z = w0 + w1·X + w2·Y**

In order to make it easy to understand the so-called regression hyperplane, a data visualization is provided. Due to the huge amount of data (53000 samples), a randomized sample of 500 samples is selected to perform the Linear Regression. This randomized sample is also plotted to the following graph:

<img src="dataviz.PNG">