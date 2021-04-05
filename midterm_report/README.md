# Trend Analysis for Cryptocurrencies and other Assets. (Team 6) 

### Introduction/Background: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cryptocurrency is drawing more and more attention from investors as more people are interested in decentralized finance. Predicting the trend of assets plays an important role in traders’ decision to buy or sell. There have been many studies on using machine learning techniques to predict the prices of Bitcoin. For example, Mallqui & Fernandes found the Support Vector Machines (SVM) algorithm performed best in forecasting the Bitcoin exchange rates, while the combination of Recurrent Neural Networks and a Tree classifier performed best in predicting the Bitcoin price direction (2019). Another study also found that SVM algorithm is a reliable forecasting model for cryptocurrency (Hitam& Ismail, 2018). 

### Problem definition: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We found several limitations in current literature. Firstly, most of them only predict price of assets, without any indication for traders whether to buy, sell or hold their investments. Also, cryptocurrency short-term predictability is difficult, thus day trading cryptocurrencies might be challenging (Liew, Li, Budavári, & Sharma, 2019). We believe that predicting trend is a better measure than predicting price.  

Secondly, no studies have employed technical indicators commonly used by stock traders in their cryptocurrency price prediction model. As Mallqui & Fernades (2019) pointed out, the technical indicators such as Relative Strength Index (RSI), Moving Average Convergence/Divergence (MACD), etc. could be used in addition to economic indicators to better predict Bitcoin price direction. Therefore, we would like to propose a model that incorporates the indicators and predicts trend to help traders decide when to sell, hold, or buy cryptocurrencies at a given moment. 

### Data Collection
Our data will be obtained from Binance using its API. We are planning to work on 5 most popular coins and our data dates to 2018.

### Data Preprocessing
In this project, we plan to train two classifiers, one for the `BUY` action and the other for the `SELL` action.
For training supervised learning ML models, we need **truth labels (`y`)** and **features (`X`)**. Below, we explain
how we generate truth labels and features from raw price data.

#### Defining & Generating truth labels (`y`)
* **`Buy trend`** (`y` for training **BUY** classifier)
  * Inspect last 5 timeframes, and the `y` is TRUE if current frame price is higher than all of the 5 last timeframes.
* **`Sell trend`** (`y` for training **SELL** classifier)
  * TODO
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

* Follow-up Questions:
  * Do we need to normalize the price/features before feeding it to our ML models?

### Methods:

#### ARIMA & GARCH (Statistical Time Series Analysis)
* Results & Discussion
  * TODO

#### Decision Tree (Supervised Learning)
* Used [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

* Results & Discussion
  * TODO

#### Linear Models (Supervised Learning)
* Ridge Regression
  * Used [`sklearn.linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn-linear-model-ridgeclassifier)
    * Converts the target `y` into {-1, 1} and then treats the problem as a regression task.

  * Results & Discussion
    * TODO

* Logistic Regression
  * Used [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)
  * Results & Discussion
    * TODO

#### Support Vector Machine (Supervised Learning)
* Results & Discussion
  * TODO

#### Q-Learning (Reinforcement Learning)
* Introduction & Explanation of how Q-Learning works?
* Results & Discussion
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

 