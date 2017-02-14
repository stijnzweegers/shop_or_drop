# **Shop or Drop**
## *Future Buying Behavior and Customer Life-Time Value*

######Table of Contents
* Motivation
* Approach
* Results
*
*
<br>

#####Motivation
Companies have high interest in predicting the future purchasing patterns of their customers, one of the reasons is the calculate their customer's future value. We can use a customer's future value to allocate our marketing budget efficiently, by targeting our customers that will yield the most profit to our company.  

For my capstone project, I'm trying to predict a customer's future life-time value, only using transaction data in a non-contractual business setting. That is, I'm trying to predict the future number of transactions and the value of those transactions for every single customer, solely using the customer id, date of transaction and value of the transaction.

Using transaction data is unconventional in the modern machine learning society, where the trend is to use as many features as possible. However, by focussing on the most important features we can already get good results and reduces noise. In addition, most companies have transaction data available, making this project widely applicable.
<br>

#####Approach
My project replicates the study from P. Fader et al., "Counting Your Customers" (2003) and future customer life-time valuation work. Peter Fader describes the beta-geometric negative binomial distribution model (BG/ NBD), which is based on the Pareto/ NBD model but claimed to be computationally less expensive. My data consists of customer transactions from fashion retail companies that sell through an e-commerce platform, such as Shopify.

From the transaction data we can generate the following features, summarized by customer:
  1. Frequency of repeated transactions
  2. Recency (when the last transaction occurred)
  3. Time a customer had the opportunity to make a repeated transaction
  4. Mean value of transaction

The BG/ NBD model works in a two-step approach. First it calculates the probability the customer is alive with a beta geometric distribution. Every customer is assumed to become inactive at one point in their live, which is referred to as the dropout-rate. Secondly, the negative binomial distribution calculates the purchasing rate for all the customers that are expected to be alive. The purchasing rate is calculated at an individual level with a Poisson distribution and on a population level with a gamma distribution.  
<br>

#####Frequency & Recency
Below we can see the the distribution for a single customer over time as an example. The customer made five repeated purchases over time and we can see how his/ her probability of being alive decreases when he/ she hasn't made a transaction for some time. If we look closely, we can see that the rate at which this customer is assumed to become inactive increases after more repeated transactions.

![Single distribution](/img/single_distribution.png)

<br>
For the total population, the relationship between recency and frequency looks as follows. Where we can see the high-frequency buying customers on the right hand side of the graph and the low-frequency buying customers on the left hand side. We can see that a customer's probability of being alive decreases when he/ she hasn't purchased something for a while. If a customer has a high-frequency buying behavior, he/ she is assumed to become inactive quicker.

![Recency frequency population](/img/rec_freq_population.png)

<br>


#####Number of Transactions
Below we can see how well the BG/ NBD model predicts the total cumulative number of transactions over time. We are training on 39 weeks and predicting on the following 39 weeks, similar to the study from P. Fader. As a comparison I used the average retention rate times the historical number of transactions, which is the grey line. The green line is the BG/ NBD base line, which does a better job predicting but is still not great.
Looking into the data it becomes clear that more than 90% of the customers only made one transaction in the training data. These customers have a different behavior than the high-frequency buying customers. Since the BG/ NBD model predicts a population wide purchasing rate, it has a hard time both predicting the low-frequency buying customers and high-frequency buying customers. Because of this reason undersampling and oversampling did not improve the model. However, splitting the training data into high-frequency and low-frequency customers and training on both groups individually did improve the model. The red line shows the result.


As a base-case I've used the average retention rate for my dataset.



The goal is to use several models on my data and see which model performs best, where the main goal is to predict churn. And predict the future transactions.
After that, I would like to look into the customer value, i.e. the amount of revenue a customer is generating. I could for example split the customers into high-value customers and into low-value customers. This would imply that high-value customers that are churning should be on the radar for the companies.
Finally, I could make some kind of an interface that gives a weekly report on customer’s probability to churn etc. This would make it easy for a company to review on the weekly basis what customers are most likely to churn and companies could then decide if they’d like to take action (by giving discounts etc.) to these specific customers.
