# Using Linear models for **stock price trend prediction**

## What data are we using?
* hour-scale price data of `BTCUSD`
  * Fetched using Binance API

## Training a classifier for the task
Train two classifiers, one for the **BUY** action and the other for the **SELL** action

### What are we trying to predict? (what is the `y`?)
* **`Buy trend`** (`y` signal when training classifiers for **BUY**)
  * Inspect last 5 timeframes, and the `y` is TRUE if current frame price is higher than all of the 5 last timeframes.
* **`Sell trend`** (`y` signal when training classifiers for **SELL**)
  * **TODO**
* Follow-up Questions
  * Are our Buy/Sell trend signals actually helpful in gaining profit?
    * TODO: Try backtesting to find this out!

### What are the features used for training? (What is the `X`?)
* We are using **technical indicators** as features.
* Example technical indicators
  * [EMA (Exponential Moving Average)](https://www.investopedia.com/terms/e/ema.asp)
  * [MFI (Money Flow Index)](https://www.investopedia.com/terms/m/mfi.asp)
  * [RSI (Relative Strength Index)](https://www.investopedia.com/terms/r/rsi.asp)
  * More to be added & experimented..

### Linear Classifier using Ridge Regression
* Used [`sklearn.linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn-linear-model-ridgeclassifier)
  * Converts the target `y` into {-1, 1} and then treats the problem as a regression task.
  * Uses Ridge Regression

* Results & Visualizations
  * TODO

### Linear Classifier using Logistic Regression
* Used [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)
  * Logistic Regression (aka logit, MaxEnt) classifier.

* Results & Visualizations
  * TODO
  