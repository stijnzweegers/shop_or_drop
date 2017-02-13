# Shop or Drop
## Future Buying Behavior and Customer Life-time Value

######Motivation
Companies have high interest in predicting the future purchasing patterns of their customers, one of the reasons is the calculate their customer's future value. We can use a customer's future value to allocate our marketing budget efficiently, by targeting our customers that will yield the most profit to our company.  

For my capstone project, I'm trying to predict a customer's future life-time value, only using transaction data in a non-contractual business setting. That is, I'm trying to predict the future number of transactions and the value of those transactions for every single customer, solely using the customer id, date of transaction and value of the transaction.

Using transaction data is unconventional in the modern machine learning society, where the trend is to use as many features as possible. However, by focussing on the most important features we can already get good results and reduces noise. In addition, most companies have transaction data available, making this project widely applicable.


######Approach
My project replicates the study from P. Fader et al., "Counting Your Customers" (2003) and future customer life-time valuation work. Peter Fader describes the beta-geometric negative binomial distribution model (BG/ NBD), which is based on the Pareto/ NBD model but claimed to be computationally less expensive.

From the transaction data we can generate the following features, summarized by customer:
  1. Frequency of repeated transactions
  2. Recency (when the last transaction occurred)
  3. Time a customer had the opportunity to make a repeated transaction
  4. Mean value of transaction

The BG/ NBD model works in a two-step approach. First it calculates the probability the customer is alive with a beta geometric distribution. Every customer is assumed to become inactive at one point in their live, which is referred to as the dropout-rate. Secondly, the negative binomial distribution calculates the purchasing rate for all the customers that are expected to be alive. The purchasing rate is calculated at an individual level with a Poisson distribution and on a population level with a gamma distribution.  


######Results
My data consists of customer transactions from fashion retail companies that sell through an e-commerce platform, such as Shopify.

Below we can see the the distribution for a single customer over time as an example. The customer made five repeated purchases over time and we can see how his/ her probability of being alive decreases when he/ she hasn't made a transaction for some time. If we look closely, we can see that the rate at which this customer is assumed to become inactive increases after more repeated transactions.


***INCLUDE the single distributuion picture***

Looking at the total population, we can see








######Scope
The goal is to use several models on my data and see which model performs best, where the main goal is to predict churn. And predict the future transactions.
After that, I would like to look into the customer value, i.e. the amount of revenue a customer is generating. I could for example split the customers into high-value customers and into low-value customers. This would imply that high-value customers that are churning should be on the radar for the companies.
Finally, I could make some kind of an interface that gives a weekly report on customer’s probability to churn etc. This would make it easy for a company to review on the weekly basis what customers are most likely to churn and companies could then decide if they’d like to take action (by giving discounts etc.) to these specific customers.
