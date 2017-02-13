import datetime
import pandas as pd
import numpy as np

from lifetimes.utils import summary_data_from_transaction_data


df_columns = pd.read_table('../data/db_dict.txt')

def txt_to_df(directory):
    headers = df_columns['field'][df_columns['table'] == 'atk_transaction'].tolist()
    df = pd.read_table(directory, header=None, names=headers, dtype = {'customer_id': str})
    df['cnt_trans'] = 1
    df = df[df['customer_id'] != '0']
    df = df[df['customer_id'] != 'UNKOWN']
    return df


def df_transform(df, site_id=0):
    '''
    input:
    site_id = type int
    '''
    # splits df by site if site_id is provided
    if site_id != 0:
        df = df[df.site_id == site_id]

    df.sort_values(by='created',ascending=True, axis=0, inplace=True)

    # check if created column is datetime, else convert
    if isinstance(df.created.iloc[0], datetime.date) == False:
        df['created'] = pd.to_datetime(df['created'])
        df['created'] = df['created'].dt.date
    # drop non used columns
    df = df.ix[:,['customer_id', 'site_id', 'created', 'cnt_trans', 'amount_usd']]
    return df


def test_holdout(summary_train_df, holdoutratio=0.2, high_freq_split=4):
    '''
    splits the summary_train_df into a holdout test set. That we need to test our model trained model on
    '''
    pass




def train_test_split(df, first_purch_weeks, train_end_week=39, train_start_date=''):
    '''
    train_start_date, of training data, needs to by type string "yyyy-mm-dd"
    first_purch_weeks = only customers are trained that have a transactions within the first purch weeks (from train_start_date)
    train_end_week = used to calculate T and to split data into train and test, needs to be type string "yyyy-mm-dd"

    Set up raw data columns as input for the model
    - t0 = 2015-04-18
    - t = 12 weeks (similar to paper)
    - first_purch
    - T = t - first purch; or time a customer could make a repeated trans
    - x = repeated transactions (= total frequency - 1) 0 if no repeated transaction occured
    - t_x = recency, time of last transaction - time of first purchase

    Exclude any customers that dont have a purchase within the first purchase weeks
    '''
    if train_start_date == '':
        df['t0'] = df.created.min()
    else:
        df['t0'] = pd.to_datetime(train_start_date)
        df['t0'] = df['t0'].dt.date

    # split df into train and test based on transaction dates
    train_end_date = (df['t0'] + datetime.timedelta(days=train_end_week*7)).iloc[0]
    test_df = df[df.created > train_end_date]
    # create column to split test data by predicted weeks
    test_df['test_weeks'] = (test_df['created'] - train_end_date).apply(lambda x: x.days/ 7.)

    # import pdb; pdb.set_trace()

    train_df = df[df.created <= train_end_date]

    # summary_train_df = summary_data_from_transaction_data(train_df, 'customer_id', 'created', 'amount_usd', freq='D', observation_period_end=train_end_date.strftime('%Y-%m-%d'))
    #
    # create input data from train_df, counting all customer transactions and other required inputs
    train_df['first_purch'] = train_df['created']
    train_df['t_x'] = train_df['created']

    train_df.sort_values(by='created',ascending=True, axis=0, inplace=True)

    summary_train_df = train_df.groupby(['customer_id', 'site_id', 't0']).agg({'cnt_trans' : np.sum, 'first_purch' : np.min, 't_x' : np.max, 'amount_usd': np.mean}).reset_index()

    # summary_train_df, summary_holdout_df = train_holdout(summary_train_df)


    # create cohort df for customers that have transactions in the first purch weeks
    first_purch_cutoff = summary_train_df['t0'].iloc[0] + datetime.timedelta(days=first_purch_weeks*7)
    summary_train_df = summary_train_df[summary_train_df.first_purch < first_purch_cutoff]

    # convert all transactions to repeated transactions
    summary_train_df['cnt_trans'] = summary_train_df['cnt_trans'] - 1

    # T: time (days) to make repeated transactions in training period
    summary_train_df['T'] = (train_end_date - summary_train_df['first_purch']).apply(lambda x: x.days/ 7.)
    # t_x: time (days) of last transaction
    summary_train_df['t_x'] = (summary_train_df['t_x'] - summary_train_df['first_purch']).apply(lambda x: x.days/ 7.)
    summary_train_df = summary_train_df.rename(columns={'cnt_trans': 'frequency', 't_x': 'recency'})

    return summary_train_df, test_df, first_purch_weeks, train_end_date

def undersampling(train_df, threshold=0.2, split=2):
    train_df_minor = train_df[train_df.x < split]
    train_df_major = train_df[train_df.x >= split]

    undersampled_df = train_df

    ratio = len(train_df_major) / float(len(undersampled_df))
    np.random.seed(10)
    while ratio < threshold:
        drop_indices = np.random.choice(train_df_minor.index, 5, replace=False)
        undersampled_df = undersampled_df.drop(drop_indices)
        train_df_minor = train_df_minor.drop(drop_indices)
        ratio = len(train_df_major) / float(len(undersampled_df))

    return undersampled_df


def oversampling(train_df, threshold=0.15, split=3):
    train_df_minor = train_df[train_df.x < split]
    train_df_major = train_df[train_df.x >= split]

    oversampled_df = train_df
    ratio = len(train_df_major) / float(len(oversampled_df))
    np.random.seed(10)

    while ratio < threshold:
        train_df_major = oversampled_df[oversampled_df.x >= split]
        oversampled_df = oversampled_df.append(train_df_major.sample(n=5, replace=True))
        ratio = len(train_df_major) / float(len(oversampled_df))

    return oversampled_df


# def train_test_split_transactions_only(df, train_start_date='', first_purch_weeks, train_end_week=52):
#     if train_start_date == '':
#         df['t0'] = df.created.min()
#     else:
#         df['t0'] = pd.to_datetime(train_start_date)
#         df['t0'] = df['t0'].dt.date
#     # split df into train and test based on transaction dates
#     train_end_date = (df['t0'] + datetime.timedelta(days=train_end_week*7)).iloc[0]
#     test_df = df[df.created > train_end_date]
#     # create column to split test data by predicted weeks
#     test_df['test_weeks'] = (test_df['created'] - train_end_date).apply(lambda x: x.days/ 7.)
#
#     train_df = df[df.created <= train_end_date]
#     # create input data from train_df, counting all customer transactions and other required inputs
#     train_df['first_purch'] = train_df['created']
#     train_df['t_x'] = train_df['created']
#
#     train_df.sort_values(by='created',ascending=True, axis=0, inplace=True)


def split_by_freq(train_df, freq_split):
    '''
    splits train_df into high_freq and low_freq, to better train the model
    '''
    train_df_high_freq = train_df[train_df['frequency'] > freq_split]
    train_df_low_freq = train_df[train_df['frequency'] <= freq_split]
    return train_df_high_freq, train_df_low_freq


def split_test_df_by_pred_period(df, pred_weeks):
    '''
    splits df into only the transactions that we are trying to predict
    used to calculate RRS for example
    '''
    df = df[df['test_weeks'] <= pred_weeks]
    test_transactions = df.groupby(['customer_id'])['cnt_trans'].sum().reset_index()
    test_dict = test_transactions.set_index('customer_id')['cnt_trans'].to_dict()
    return test_dict




# if __name__ == '__main__':
#
#     df = txt_to_df('../data/atk_transaction.txt')
#     transformed_df = df_transform(df, 23395)
#     train_df, test_df, first_purch_weeks = train_test_split(transformed_df, first_purch_weeks=2, train_end_week=10)
    # check ID '415306753', '382184409'

    # export_df_to_csv(train_df)
