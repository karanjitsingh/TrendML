## Project Final Report 

### Introduction/Background: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cryptocurrency is drawing more and more attention from investors as more people are interested in decentralized finance. Predicting the trend of assets plays an important role in traders’ decision to buy or sell. There have been many studies on using machine learning techniques to predict the prices of Bitcoin. For example, Mallqui & Fernandes found the Support Vector Machines (SVM) algorithm performed best in forecasting the Bitcoin exchange rates, while the combination of Recurrent Neural Networks and a Tree classifier performed best in predicting the Bitcoin price direction (2019). Another study also found that SVM algorithm is a reliable forecasting model for cryptocurrency (Hitam& Ismail, 2018). 

### Problem definition: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We found several limitations in current literature. Firstly, most of them only predict price of assets, without any indication for traders whether to buy, sell or hold their investments. Also, cryptocurrency short-term predictability is difficult, thus day trading cryptocurrencies might be challenging (Liew, Li, Budavári, & Sharma, 2019). We believe that predicting trend is a better measure than predicting price.  

Secondly, no studies have employed technical indicators commonly used by stock traders in their cryptocurrency price prediction model. As Mallqui & Fernades (2019) pointed out, the technical indicators such as Relative Strength Index (RSI), Moving Average Convergence/Divergence (MACD), etc. could be used in addition to economic indicators to better predict Bitcoin price direction. Therefore, we would like to propose a model that incorporates the indicators and predicts trend to help traders decide when to sell, hold, or buy cryptocurrencies at a given moment. 

### Data Collection
Our data is obtained from Binance using its API. We are planning to work on 5 most popular coins and our data dates to 2018.

### Data Preprocessing
In this project, we plan to train two classifiers, one for the `BUY` action and the other for the `SELL` action.
For training supervised learning ML models, we need **truth labels (`y`)** and **features (`X`)**. Below, we explain
how we generate truth labels and features from raw price data.

#### Defining & Generating truth labels (`y`)
* **`Buy trend`** (`y` for training **BUY** classifier)
  * Inspect the next 5 timeframes, and the `y` is TRUE if the closing price of any of those frames is higher than the current frame by 5% or more.
* **`Sell trend`** (`y` for training **SELL** classifier)
  * Inspect the next 5 timeframes, and the `y` is TRUE if the closing price of any of those frames is lower than the current frame by 5% or more.
* Follow-up Questions
  * Are our Buy/Sell trend signals justifiable? How much are they helpful in gaining actual profit? 

#### Defining & Generating features (`X`)
* We use **technical indicators** as features.
* We use [`TA-Lib`](https://www.ta-lib.org/) to calculate technical indicators from raw price data.
* Example technical indicators
  * [EMA (Exponential Moving Average)](https://www.investopedia.com/terms/e/ema.asp)
  * [MFI (Money Flow Index)](https://www.investopedia.com/terms/m/mfi.asp)
  * [RSI (Relative Strength Index)](https://www.investopedia.com/terms/r/rsi.asp)
  * etc..



### ARIMA & GARCH (Statistical Time Series Analysis)
#### Implementation 
In order to evaluate the ARIMA model, we used the error function called Symmetric Mean Absolute Percentage Error (SMAPE), which is commonly used as an accuracy measure based on relative errors. We obtained a SMAPE of 56.815.

We used p=5, d=1 and q=0 as the ARIMA parameters for our implementation

#### Result and discussion
![](https://github.com/karanjitsingh/untitled/blob/master/Checkpoints/Final/arima1.png)

This is the zoomed in version of the above graph.
![](https://github.com/karanjitsingh/untitled/blob/master/Checkpoints/Final/arima2.png)

We could see that the result from this model offers a pretty good prediction accuracy and the training process is relatively fast.
Note that this model only predicts prices and doesn’t predict trend. Therefore, we also tried different models that can achieve our objective of predicting trend.


### Decision Tree (Supervised Learning)
* **Used** [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

* **Method**  
  Decision tree is a simple but effective machine learning technique that can be used for classification tasks. In this project, We use it for the purpose of selecting the best   set of features among several available features. In Technical Analysis, different approaches make use of different indicators to observe a pattern and determine the stock       trends. More often than not, a set of 2-3 indicators are used in combination rather than just one for better accuracy. It is a challenge to identify which set performs better,   so we make use of decision trees to identify them for us. We have considered 8 commonly used indicators ( SMA21, SMA50, EMA21, EMA50, RSI, MFI, ADX, ATR ) as features. A set     of 3 among them are selected randomly and provided as features to the decision tree. The classification is done separately for 'BUY' or 'Long' and 'SELL' or 'Short' based on     the truth labels as defined in sec 4.1. The training and test data split is handled by the sklearn library and the resulting accuracy is calculated. This process is repeated     several times considering a different set of indicators again selected randomly from the pool. The indicator set with the highest accuracy is selected as the best or the most   relavant indicators to be used for the asset.

* **Results and Discussion**    
   The graph below plots the selected indicator sets on the x axis with the obtained accuracy in the y axis. We can observe that the combination 8-7-2 or (ATR, ADX and SMA50)       provides the best estimate of the trend among the selected indicators as it has the highest accuracy.
  
  ![image](https://user-images.githubusercontent.com/48078197/113922177-f7aab780-97b4-11eb-87ac-7a8839c21f92.png)
  
  The below image shows the branch traversal of the decision tree to reach the classification or the leaf nodes. It is important to note that the max depth of the tree has been   selected as 3 to limit overfitting as technical indicators are very prone to overfit the data in stock trend and price prediction. To give us a better evidence and further       control overfitting, another constraint has been placed where each leaf is considered valid only if there are atleast 25 samples in it.
  
  ![image](https://user-images.githubusercontent.com/48078197/113923315-6b00f900-97b6-11eb-84e1-636756fec762.png)
  
  The gini index is a metric used to measure the purity of the node. It is similar to entropy in use and can be used to observe the quality of the split.  
  GiniIndex = 1&nbsp; –&nbsp; $\sum_{i=1}^n$ $p^2_i$ 
  
  By printing out the decision tree, we can get a better understanding of how the split and decision is done at each node and how each indicator is being considered. For the       above selection, the tree is represented as -  

  ```
  1: 'sma21', 2: 'sma50', 3: 'ema21', 4: 'ema50', 5: 'rsi14', 6: 'mfi10', 7:'adx', 8:'atr' 
  |--- 2 <= 10808.51  
  ||--- 8 <= 1084.19  
  |||--- 2 <= 5198.32  
  ||||--- class: False  
  |||--- 2 >  5198.32  
  ||||--- class: False  
  ||--- 8 >  1084.19  
  |||--- 8 <= 3901.93  
  ||||--- class: True  
  |||--- 8 >  3901.93  
  ||||--- class: True  
  |--- 2 >  10808.51  
  ||--- 8 <= 1208.95  
  |||--- 7 <= 28.66  
  ||||--- class: True  
  |||--- 7 >  28.66  
  ||||--- class: True  
  ||--- 8 >  1208.95  
  |||--- 2 <= 29431.01  
  ||||--- class: False  
  |||--- 2 >  29431.01  
  ||||--- class: True  
  
  classification accuracy for atr, adx, sma50 : 0.7161290322580646 
  ```

By changing the tree parameters and constraints such as max depth, minimum nodes and trend thresholds, we can obtain a better picture of the selected indicators and it can be a significant help for traders in identifying the right indicators to use in their trading. The focus here is not on the absolute metric of accuracy but rather the relative measure of accuracy among different indicator sets as shown below:   
**Indicator sets**: ['4-8-1', '4-7-5', '3-6-7', '8-7-2', '8-2-3', '7-1-8', '3-5-4', '7-4-8']   
**Accuracy**      : [66.45,    65.80,    63.22,   71.61,  66.77,   59.35,   68.38,  70.96]  
<br>
By using the backtrader library in python, we were able to test the results from the above decision tree. By using the indicators and their respective conditions provided by the tree as a rule for our buy and sell signal, starting with a capital of $10000, we were able to observe if the different sets and the corresponding final values were in line with expectations. For 3 years of data starting from 2017 to 2020, the indicator set with the highest accuracy, i.e. 8-7-2 ended with a net value of $16472 (profit) whereas a different set of indicators with lower confidence ended with a net value of $9424 (loss). While the backtrader is not a comprehensive strategy and the absolute percentage of profit or loss may not be very reliable, it provides a very good estimate of how good or bad a given set of technical indicators are in determining the trend for the underlying asset. This information is often very crucial for a trader in determining their trades, direction and position size.

### Linear Models (Supervised Learning)
Before diving into more sophisticated models, we ran initial experiments with linear models which tend to have relatively simple decision boundaries. The motivation for trying out the linear models was to see how much classification accuracy the simple linear models can achieve on our data. We chose to try ridge regression & logistic regression models for our classification task.

* RidgeClassifier (using Ridge Regression)
  * Used [`sklearn.linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn-linear-model-ridgeclassifier), which converts the target `y` labels into {-1, 1} and then treats the classification problem as a regression task.

* Logistic Regression
  * Used [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)

Currently, we have only trained classifiers that return the `BUY` trend signal. We have also only used 3 technical indicators (EMA-30, MFI-10, RSI-30) as features for training.

We initially trained the models on two datasets derived from different timeframe units of `BTCUSDT` price data. 
* X, y derived from **1hour-scale price data**
  * Ratio of True/False `y` labels: 24655 / 6013
* X, y derived from **1day-scale price data**
  * Ratio of True/False `y` labels: 1039 / 239

Note that the True/False ratio of `y` in the datasets are unbalanced, and the number of `True` labels is more than 4 times the number of `False` labels. Below, we report our results from training the two on the original unbalanced datasets.

| Model (trained on 1hour-scale data)     | training set accuracy | test set accuracy |
| -----------| --------------------- | ----------------- |
| Ridge Classifier| 0.8057647929638176| 0.8053814002089864|
| Logistic Regression | 0.8042844080637437 | 0.8035527690700105 |

| Model (trained on 1day-scale data)     | training set accuracy | test set accuracy |
| -----------| --------------------- | ----------------- |
| Ridge Classifier| 0.8062770562770563| 0.8284789644012945|
| Logistic Regression | 0.8062770562770563 | 0.8252427184466019 |

<br>

The surprisingly high accuracy of 80% was achievable due to the unbalanced dataset that we were using. Below is a plot showing the distribution of true labels and the label predictions from our Ridge Regresesion model on 1-day scale data. **It is notable that our model always returned `True` for the classification task, and yet its accuracy exceeded 80% thanks to fact that 80% of the dataset had the label `y = True`.**

 ![image](https://github.com/karanjitsingh/untitled/blob/master/Checkpoints/Midterm/assets/ridge_skewed_1d.png)

In order to address the **unbalanced dataset problem**, we truncated the original dataset by randomly removing some of the data points with `y = True`. This truncating process balanced the distribution of `y` . Below are descriptions on the truncated datasets.

* X, y derived from **1hour-scale price data** (truncated)
  * Ratio of True/False `y` labels: 6002 / 6002
* X, y derived from **1day-scale price data** (truncated)
  * Ratio of True/False `y` labels: 233 / 233

We trained the two linear models on the truncated datasets, and the results
were as below.

| Model (trained on truncated 1hour-scale data)     | training set accuracy | test set accuracy |
| -----------| --------------------- | ----------------- |
| Ridge Classifier| 0.657336443407753| 0.6491169610129957|
| Logistic Regression | 0.6578918138398312 | 0.647450849716761 |

| Model (trained on truncated 1day-scale data)     | training set accuracy | test set accuracy |
| -----------| --------------------- | ----------------- |
| Ridge Classifier| 0.7392550143266475| 0.6752136752136753|
| Logistic Regression | 0.7020057306590258 | 0.6495726495726496 |

<br>

The linear models trained on the truncated datasets exhibit test set accuracy around 65%. Considering the given dataset has 50/50 ratio of True/False for the `y` label, we can say that the linear models succeeded in finding a decision boundary that has better classification accuracy than a coin-toss.

<br>

### Support Vector Model 
After exploring the linear regression models, we tried another supervised machine learning model - support vector machine (SVM). We used SVM's linear and non-linear radial kernal for classification. We used the cleaned balanced data (50/50). We checked the accuracy of the SVM model on various C Parameter with gamma value set to auto. C Parameter tells the SVM optimization how much mis-classification you want to avoid on each training example. For large values of C, the optimization chooses a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. As large values of C could lead to overfitting and small values lead to lower accuracy, We picked C =1 as an optimum value.


Table 1: SVM Radial(R) and Linear(L) Model Trained on truncated 1d-scale data

| Model   | C Value |  Train Accuracy | Test Accuracy |
|---------|---------|-----------------|---------------|
| L       | 1e-3    | 0.5143          | 0.5139        |
| R       | 1e-3    | 0.5043          | 0.5043        |
| L       | 1e-2    | 0.6741          | 0.6676        |
| R       | 1e-2    | 0.5043          | 0.5043        |
| L       | 1e-1    | 0.7006          | 0.6762        |
| R       | 1e-1    | 0.6991          | 0.6733        |
| L       | 1       | 0.7020          | 0.6733        |
| R       | 1       | 0.7077          | 0.6762        |
| L       | 1e1     | 0.7027          | 0.6733        |
| R       | 1e1     | 0.7221          | 0.6705        |
| L       | 1e2     | 0.7027          | 0.6733        |
| R       | 1e2     | 0.7407          | 0.6360        |
| L       | 1e3     | 0.7027          | 0.6733        |
| R       | 1e3     | 0.7679          | 0.6102        |

The SVM model trained on the truncated datasets provide the best score of around 68% on test dataset with C=1 and a radial kernel. 
<br>

### Q-Learning (Reinforcement Learning)
#### Introduction & Explanation of how Q-Learning works?
In Reinforcement Learning, we will have an agent that will take in a state of an environment (s), then look up the policy (Pi) on what it should do and output an action (a). There will be reward (r) associated with an action that the agent decides to take. If the action changes the environment, then we will have a new state of an environment then the circle repeats again, the agent will look up the policy and output an action. The objective of the agent is to take actions that optimize the reward over time.
In the context of trading, actions include buy, sell, and hold. Reward could be return from trade or daily return. States are factors about our assets(stocks/ cryptocurrencies) that we might observe and know about like prices.
This is also called Markov decision problem which include:
*  set of states s
*  set of actions a 
*  transition function T[s,a,s']: a three-dimensional object that records the probability that if we are at state s and take action a we will end up at state s'
*  reward function R[s,a] 
We need to find $\Pi$ (s) that will maximize the reward over time.
Most of the time, we don't know the transition or/ and the reward function, so our agent has to interact with the world and observe what happens and learn from it. We call those experience tuples. Once we have these experience tuples, we will use Q-learning, a model-free method, to develop a policy $ \Pi $ just by directly looking at our data.


In Q-learning, we want to optimize discounted reward: 

![formula](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D0%7D%5E%7B%5Cinfty%7D%20%5Cgamma%5E%7Bi-1%7Dr_i%5Chspace%7B30pt%7Ds.t.%5Chspace%7B5pt%7D0%20%5Cleq%20%5Cgamma%5Cleq1)

Gamma (γ) is strongly related to interest rate. For example, if gamma = 0.95, it means each step in the future is worth 5% less than the immediate reward if we got it right away.


In Q-learning, we will have a Q table that represents the value of taking action a in state s.
* ` Q[s,a] = immediate reward + discounted reward`

When we are in a state s and we want to find out which action is the best to take, we need to look at all potential actions and find out which value of Q[s,a] is maximized, so our policy is represented as :
 `Pi(s) = argmax_a(Q[s,a])`

Our optimal policy and optimal Q-table are represented as `Pi*(s) and Q*[s,a]`.

Q-learning procedure
-	Select training data
-	Iterate over time <s,a,s’,s>
-	Test policy pi
-	Repeat until converge


Details of Iterate over time <s,a,s’,s>:
-	Set start time, init Q[] (small, random number)
-	Compute s (prices of our cryptocurrencies)
-	Consult Q to find the best option a
-	Select a
-	Observe r, s’
-	Update Q


Update Rule: 
The formula for computing Q for any state-action pair <s, a>, given an experience tuple <s, a, s', r>, is:
`Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])`
Here:
-	r = R[s, a] is the immediate reward for taking action a in state s,
-	γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards,
-	s' is the resulting next state,
-	argmaxa'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s', and,
-	α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.

Recap:

Building a model:
* Define states, actions, rewards
* Choose in-sample training period
* Iterate: Q-table update
* Backtest


Testing a model:
* Backtest on later data


Advantages: Q-learning can easily be applied to domains where all states and/or transitions are not fully defined


Challenges: reward (e.g. for buying a stock) often comes in the future - representing that properly requires look-ahead and careful weighting, taking random actions (such as trades) just to learn a good strategy is not really feasible because it will cost us lots of money (Reference: Udacity).


#### Results & Discussion
  * TODO
______

#### Contributions
* Himanshu Gupta : 
* Karan Singh : 
* Tien Le : 
* Vikas Rao : 
* Youngsuk Kim : 

#### References 

[Akyildirim, E., Goncu, A., & Sensoy, A. (2020). Prediction of cryptocurrency returns using machine learning. Annals of Operations Research, 297(1-2), 3-36. doi:10.1007/s10479-020-03575-y](https://www.researchgate.net/publication/340500312_Prediction_of_cryptocurrency_returns_using_machine_learning)

[Bu, S., & Cho, S. (2018). Learning optimal q-function using deep boltzmann machine for reliable trading of cryptocurrency. Intelligent Data Engineering and Automated Learning – IDEAL 2018, 468-480. doi:10.1007/978-3-030-03493-1_49](https://ouci.dntb.gov.ua/en/works/7P22RGxl/)

[Hitam, N. A., & Ismail, A. R. (2018). Comparative performance of machine learning algorithms for cryptocurrency forecasting. Indonesian Journal of Electrical Engineering and Computer Science, 11(3), 1121. doi:10.11591/ijeecs.v11.i3.pp1121-1128](https://www.researchgate.net/publication/326837070_Comparative_Performance_of_Machine_Learning_Algorithms_for_Cryptocurrency_Forecasting)

[Mallqui, D. C., & Fernandes, R. A. (2019). Predicting the DIRECTION, maximum, minimum and closing prices of daily Bitcoin exchange rate using machine learning techniques. Applied Soft Computing, 75, 596-606. doi:10.1016/j.asoc.2018.11.038](https://www.sciencedirect.com/science/article/pii/S1568494618306707)

[Liew, J., Li, R., Budavári, T., & Sharma, A. (2019). Cryptocurrency investing examined. The Journal of the British Blockchain Association, 2(2), 1-12. doi:10.31585/jbba-2-2-(2)2019](https://www.researchgate.net/publication/337011389_Cryptocurrency_Investing_Examined)
