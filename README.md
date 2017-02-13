# CapStone-project-churn
##Predicting Churn and customer life-time value

######Motivation
Companies have high interest in the customer lifetime value. In general, it’s cheaper for a company to prevent customers to churn than it is to acquire new customers. Besides churn, companies are also interested in how frequently a customer is going to make a purchase in the future, and related, how much they are expected to spend. 

Customer lifetime value can be measured in several ways, some methods are more important than others depending to a company, depending on the nature of the business. For example, a subscription-based company would be mainly interested in churn and a web shop would be more interested in how much a customer spends (total basket). 
With the increase of data and the development in algorithms, data scientists are increasingly better able to predict customer lifetime value Predicting volumes, revenue and churn is therefore shifting accordingly from business acumen people to data driven people. 


######Techniques and pipeline
My data consists of customer transactions from companies that sell through an e-commerce platform (e.g. Amazon or Shopify). From this data I can retrieve how many transactions a customer made in a specific time period (i.e. frequency) and I lookup when the last transaction occurred (i.e. recency). Also the transaction amount is included in the data. 
Based on these variables, I would like to predict:

1. Churn (whether a customer is likely to return)
2. Number of future transactions (if not churn)
3. The expected value of future transactions

From the outcome of the above 3 items, I would like to make a recommendation on what the next steps could be for that company. For example how they can maintain their customers and on what customers they should focus. 
P. Fader et. al. (2003), predicts the future transactions using the beta-geometric/ Negative Binominal Distribution (BG/ NBD) model described in the paper “Counting Your Customers” the Easy Way: An Alternative to the Pareto/ NBD Model’. 


######Scope
The goal is to use several models on my data and see which model performs best, where the main goal is to predict churn. And predict the future transactions. 
After that, I would like to look into the customer value, i.e. the amount of revenue a customer is generating. I could for example split the customers into high-value customers and into low-value customers. This would imply that high-value customers that are churning should be on the radar for the companies. 
Finally, I could make some kind of an interface that gives a weekly report on customer’s probability to churn etc. This would make it easy for a company to review on the weekly basis what customers are most likely to churn and companies could then decide if they’d like to take action (by giving discounts etc.) to these specific customers. 


# shop_or_drop
