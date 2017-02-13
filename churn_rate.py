import pandas as pd
import numpy as np
import pickle
import datetime

from convert_df import txt_to_df, df_transform, train_test_split, split_by_freq


# churn_threshold = 90  #in days, e.g. churn threshold = 90 means that if a customer didn't buy anything for 90 days he/she is considered to be inactive

data_source = '../data/atk_transaction.txt'
site_id = 30038

def customer_base(df, date, churn_threshold):
    '''
    get all customers ID's that made a transaction in the past 90 days
    output: customer base
    '''
    start_date = date - datetime.timedelta(days=churn_threshold)
    base = df.customer_id[(df.created > start_date) & (df.created <= date)]
    return list(set(base))


def churn_rate(customer_base, customers_lost):
    '''
    churn rate customer lost/ customer base

    Output = churn rate for a given moment in time
    '''
    customer_base = pd.Series(customer_base)
    retained = customer_base.isin(customers_lost).sum()
    lost_num = len(customer_base) - retained
    return lost_num/ float(len(customer_base))


def avg_churn_rate(df, train_end_date, train_start_date='', churn_threshold=30):
    '''
    calculate the average churn rate for every month
    start point = train_start_date + churn threshold
    train_end_date = end of training period
    Output: average churn rate
    '''
    if train_start_date == '':
        train_start_date = df.created.min()
    else:
        train_start_date = datetime.datetime.strptime(train_start_date,  "%Y-%m-%d").date()
    train_start_date = train_start_date + datetime.timedelta(days=churn_threshold*2)

    train_end_date = datetime.datetime.strptime(train_end_date,  "%Y-%m-%d").date()

    base_ids = customer_base(df, train_start_date, churn_threshold*2)
    check_ids = customer_base(df, train_end_date, churn_threshold*7)
    return churn_rate(base_ids, check_ids)
    # base_date = train_start_date
    #
    #
    #

    # churn_rates = []
    # for churn_days in np.arange(churn_threshold, train_days, churn_threshold):
    #     current_date = train_start_date + datetime.timedelta(days=churn_days)
    #
    #
    #     base_ids = customer_base(df, base_date, churn_threshold)
    #     check_ids = customer_base(df, current_date, churn_threshold)
    #     churn_rates.append(churn_rate(base_ids, check_ids))
    #
    # return sum(churn_rates)/ float(len(churn_rates))


def avg_number_of_transactions(df, train_end_date):

    train_end_date = datetime.datetime.strptime(train_end_date,  "%Y-%m-%d").date()
    train_df = df[df.created < train_end_date]

    train_df = train_df.groupby(['customer_id', 'site_id']).agg({'cnt_trans' : np.sum}).reset_index()
    # import pdb; pdb.set_trace()
    avg_trans = train_df.cnt_trans.sum()/ float(len(train_df))
    return avg_trans, train_df

# def pred_transactions(num_cust, avg_churn_rate, avg_num_trans):




def total_prediction(train_df, train_end_date, pred_weeks='', train_start_date='', churn_threshold=270):
    '''
    Make a single prediction for all train_df and return sum over a specific period of time
    returns a matrix of expected sales by customer and the total sales prediction
    '''
    if pred_weeks == '':
        pred_weeks = 39

    train_df['pred'] = 0.
    train_df['actual'] = 0.
    pred_df = train_df[['customer_id', 'cnt_trans','pred', 'actual']]
    pred_actual_matrix = pred_df.as_matrix()
    retention_rate = 1 - avg_churn_rate(transformed_df, train_end_date=train_end_date)

    total_freq = 0
    for i in range(0, train_df.shape[0]):
        ID = train_df.iloc[i]['customer_id']
        x = train_df.iloc[i]['cnt_trans']

        pred = self.single_conditional_prediction(ID, x, t_x, T , pred_weeks)

        self.pred_actual_matrix[i][1] = pred
        total_freq += pred
    return self.pred_actual_matrix, total_freq







if __name__ == '__main__':
    # df = txt_to_df(data_source)
    # print '------df loading done-----'
    # transformed_df = df_transform(df, site_id=site_id)
    # print '----transformed df done----'
    # transformed_df.to_pickle('transformed_df.pickle')
    # print '-------pickling done-------'
    transformed_df = pd.read_pickle('transformed_df.pickle')
    print avg_churn_rate(transformed_df, train_start_date='2015-04-01', train_end_date='2016-01-16')
    # print avg_number_of_transactions(transformed_df, train_end_date='2016-04-01')
    # print total_prediction(transformed_df, train_end_date='2016-04-01', pred_weeks=39,train_end_date, train_start_date='', churn_threshold=90)
