# **Shop or Drop**
## *Future Buying Behavior and Customer Life-Time Value*

**Stijn Zweegers, Galvanize Capstone Project - February 2017**

###Table of Contents
* [Motivation](#motivation)
* [Approach](#approach)
* [Frequency & Recency](#frequency--recency)
* [Number of Transactions](#number-of-transactions)
* [Customer Life-Time Value](#customer-life-time-value)
* [Conclusion](#conclusion)
* [Future Work](#future-work)
<br>

###Motivation
Companies have high interest in predicting the future purchasing patterns of their customers, one of the reasons is the calculate their customer's future value. We can use a customer's future value to allocate our marketing budget efficiently, by targeting our customers that will yield the most profit to our company.  

For my capstone project, I'm trying to predict a **customer's life-time value**, only using **transaction data** in a **non-contractual** business setting. That is, I'm trying to predict the future number of transactions and the value of those transactions for every single customer, solely using the customer id, date of transaction and value of the transaction.

Using transaction data is unconventional in the modern machine learning society, where the trend is to use as many features as possible. However, by focussing on the most important features we can already get good results and reduces noise. In addition, most companies have transaction data available, making this project widely applicable.
<br>

###Approach
My project replicates the study from **P. Fader et al., "Counting Your Customers" (2003)** and future customer life-time valuation work. Peter Fader describes the **beta-geometric negative binomial distribution model (BG/ NBD)**, which is based on the Pareto/ NBD model but claimed to be computationally less expensive. My data consists of customer transactions from fashion retail companies that sell through an e-commerce platform, such as Shopify.

From the transaction data we can generate the following features, summarized by customer:
  1. Frequency of repeated transactions
  2. Recency (when the last transaction occurred)
  3. Time a customer had the opportunity to make a repeated transaction
  4. Mean value of transaction

The BG/ NBD model works in a two-step approach. First it calculates the probability the customer is alive with a beta geometric distribution. Every customer is assumed to become inactive at one point in their live, which is referred to as the dropout-rate. Secondly, the negative binomial distribution calculates the purchasing rate for all the customers that are expected to be alive. The purchasing rate is calculated at an individual level with a Poisson distribution and on a population level with a gamma distribution.  
<br>

###Frequency & Recency
Below we can see the the distribution for a **single customer** over time as an example. The customer made five repeated purchases over time and we can see how his/ her probability of being alive decreases when he/ she hasn't made a transaction for some time. If we look closely, we can see that the rate at which this customer is assumed to become inactive increases after more repeated transactions.

![Single distribution](/img/single_distribution.png)

<br>
For the total **population**, the relationship between **recency and frequency** looks as follows. Where we can see the high-frequency buying customers on the right hand side of the graph and the low-frequency buying customers on the left hand side. We can see that a customer's probability of being alive decreases when he/ she hasn't purchased something for a while. If a customer has a high-frequency buying behavior, he/ she is assumed to become inactive quicker.

![Recency frequency population](/img/rec_freq_population.png)

<br>


###Number of Transactions
Below we can see how well the BG/ NBD model predicts **the total cumulative number of transactions over time**. The BG/ NBD model includes the probability of a customer being alive as a condition, and given this probability making a prediction on the future number of transactions.

I trained on 39 weeks and predicting on the following 39 weeks, similar to the study from P. Fader. As a comparison I used the average retention rate times the historical number of transactions, which is the grey line. The green line shows the BG/ NBD base line, which does a better job predicting the total number of transactions but is still not great.

Looking into the data it becomes clear that more than 90% of the customers only made one transaction in the training data. These customers have a different behavior than the high-frequency buying customers. Since the BG/ NBD model creates one population-wide purchasing rate distribution, it has a hard time both predicting the low-frequency buying customers and high-frequency buying customers. Because of this reason undersampling and oversampling did not improve the model. However, **splitting the training data into high-frequency and low-frequency customers** and training on both groups individually did improve the model, showing by the red line in the plot.

![Cumulative number of transactions over time](/img/cum_num_trans.png)

The mean absolute error (MAE) is easy interpretable to tell us how far we are off for each customer. **The MAE is 0.366** for the high/ low frequency customer's BG/ NBD model at 39 weeks, where the BG/ NBD model MAE is 0.402 or an improvement of 8.5%. To confirm, I also tested the model on a different company's dataset, which showed similar results (MAE of 0.521).

Alternatively, I've tested the PARETO/ NBD model where the BG/ NBD model is a derivative from and the Modified BG/ NBD model, which does not assume a customer's probability of being alive is 100% after one transaction, but both did not improve the BG/ NBD model.
<br>

###Customer Life-Time Value
Now that we know that we can predict the number of transactions over time pretty accurately, we can add the value of the transactions into the equation, to get a single valuation for each customer. We can do this with **the Gamma-Gamma model**. The Gamma-Gamma model is an extension for the BG/ NBD model and includes both the customer's individual historical average value of transactions as well as the population's average number of transactions. we can only apply the Gamma-Gamma model whenever there is **no significant correlation between the monetary value and the frequency of the transactions**. The table below confirms that there is no apparent correlation.

![CLV](/img/correlation_mon_val.png)
<br>

Given that there is no significant correlation between monetary value and frequency we can use the Gamma-Gamma model. Below we can see **the expected number of transactions and the conditional expected average value by customer plotted at 39 weeks**.

![CLV](/img/CLV.png)

<br>


This overview **helps us to segment our customers** by customer value based on how many transactions we can expect them to make and the value of those transactions. I've selected a threshold of frequency 2 and value of transaction of 100, however these thresholds are company specific and may vary depending on the nature of the business. The important take away is that we have a clear overview on who our most valuable customers are and how much they are worth to us, that is we know how much we can spend on them individually.

To express the performance in dollars we can look at the MAE. The MAE for future customer value is $12, where the MAE for the average* is $22.
<br>

###Conclusion
The BG/ NBD model that was described by P. Fader also worked well on my e-commerce dataset. It's an improvement to the traditional customer life-time valuation, since it provides us a single valuation by customer.

The model is widely applicable due to the fact that you only need transactional data to implement, which most companies have available.
<br>

* *average retention rate times the average value of historical transactions by customer*

##Future Work
* Include seasonality; <br>
The BG/ NBD model does currently not include seasonality - therefore adding seasonality would improve the model for companies that have high seasonality.
* Include new customers; <br>
The BG/ NBD model does currently only make predictions for the existing customers. Adding new customers to the prediction would complete the picture, especially for companies that have a high portion of their revenue coming from new customers.
