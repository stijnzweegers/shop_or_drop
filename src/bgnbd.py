"""
Implementation of the beta-geometric/NBD (BG/NBD) model from '"Counting Your train_df" the Easy Way: An Alternative to
the Pareto/NBD Model' (Fader, Hardie and Lee 2005) http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf and
accompanying technical note http://www.brucehardie.com/notes/004/
Apache 2 License
"""
from math import log, exp, sqrt

import numpy as np
import pandas as pd
import pickle
import datetime

from scipy.optimize import minimize
from scipy.special import gammaln

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from lifetimes import BetaGeoFitter
from lifetimes import ModifiedBetaGeoFitter
from lifetimes import ParetoNBDFitter
# from lifetimes import BetaGeoBetaBinomFitter

from convert_df import split_test_df_by_pred_period, txt_to_df, df_transform, train_test_split

__author__ = 'Stijn Zweegers'


'''
To do:
- Change input format
- Add conditional expectation
- Convert functions to a class
'''

class BgNbd(object):
    '''
    class for creating BG/ NBD for non contractual transactions,
    takes a csv file as input with the following columns: customer ID, number of transactions x, time of last transaction, time customer could make a repeated transaction (T)
    '''

    def __init__(self, train_df, first_purch_weeks=25, train_weeks=52, max_iter=150):
        self.train_weeks = train_weeks
        self.train_df = train_df.reset_index()
        self.first_purch_weeks = first_purch_weeks
        self.max_iter = max_iter


    def log_likelihood_individual(self, r, alpha, a, b, x, tx, t):
        """Log of the likelihood function for a given randomly chosen individual with purchase history = (x, tx, t) where
        x is the number of transactions in time period (0, t] and tx (0 < tx <= t) is the time of the last transaction - sheet BGNBD Estimation"""

        ln_a1 = gammaln(r + x) - gammaln(r) + r * log(alpha)
        ln_a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        ln_a3 = -(r + x) * log(alpha + t)
        a4 = 0
        if x > 0:
            a4 = exp(log(a) - log(b + x - 1) - (r + x) * log(alpha + tx))
        return ln_a1 + ln_a2 + log(exp(ln_a3) + a4)


    def log_likelihood(self, r, alpha, a, b, train_df):
        """Returns sum of the individual log likelihoods
        equal to cell B5 in sheet BGNBD Estimation"""
        # can't put constraints on n-m minimizer so fake them here
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return -np.inf
        train_df['ll'] = train_df.apply(lambda row: self.log_likelihood_individual(r, alpha, a, b, row['frequency'], row['recency'], row['T']), axis=1)
        return train_df.ll.sum()


    def maximize(self, train_df):
        '''Minimize (negative of maximizing) the log_likelihood function,
        i.e. change r, alpha, a and b so that the sum of indiviual log likelihoods is maximized'''
        negative_ll = lambda params: -self.log_likelihood(*params, train_df=train_df)
        params0 = np.array([1., 1., 1., 1.])
        res = minimize(negative_ll, params0, method='nelder-mead', options={'xtol': 1e-4})
        return res


    def lifetimes_fit(self, train_df, penalizer_coef=0.0):
        '''
        fit but then using lifetimes library --> faster
        '''
        bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        data = train_df
        bgf.fit(data['frequency'], data['recency'], data['T'])
        params = bgf.params_
        self.r = params['r']
        self.alpha = params['alpha']
        self.a = params['a']
        self.b = params['b']
        return params

    def modified_fit(self, train_df, penalizer_coef=0.0):
        '''
        Modified BetaGeometric NBD model from lifetimes
        '''
        mbgf = ModifiedBetaGeoFitter(penalizer_coef=penalizer_coef)
        mbgf.fit(train_df['frequency'], train_df['recency'], train_df['T'])
        params = mbgf.params_
        self.r = params['r']
        self.alpha = params['alpha']
        self.a = params['a']
        self.b = params['b']
        return params

    def pareto_fit(self, train_df, penalizer_coef=0.0):
        '''
        The difference between the models is that the BG/BB is used to describe situations in which customers have discrete transaction opportunities, rather than being able to make transactions at any time
        '''
        pareto = ParetoNBDFitter()
        data = train_df
        pareto.fit(data['frequency'], data['recency'], data['T'])
        params = pareto.params_
        self.r = params['r']
        self.alpha = params['alpha']
        self.s = params['s']
        self.beta = params['beta']
        return params


    def _negative_log_likelihood(self, params, freq, rec, T, penalizer_coef):

        if np.any(np.asarray(params) <= 0.):
            return np.inf

        r, alpha, s, beta = params
        x = freq

        r_s_x = r + s + x

        A_1 = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * log(beta)
        log_A_0 = ParetoNBDFitter._log_A_0(params, freq, rec, T)

        A_2 = np.logaddexp(-(r + x) * log(alpha + T) - s * log(beta + T), log(s) + log_A_0 - log(r_s_x))

        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -(A_1 + A_2).mean() + penalizer_term


    def pareto_conditional_expected_number_of_purchases_up_to_time(self, pred_weeks, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (frequency, recency, T)
        Parameters:
            t: a scalar or array of times.
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.
        Returns: a scalar or array
        """
        x, t_x = frequency, recency
        params = self.r, self.alpha, self.s, self.beta
        r, alpha, s, beta = self.r, self.alpha, self.s, self.beta

        likelihood = -self._negative_log_likelihood(params, x, t_x, T, 0)
        first_term = gammaln(r + x) - gammaln(r) + r*log(alpha) + s*log(beta) - (r + x)*log(alpha + T) - s*log(beta + T)
        second_term = log(r + x) + log(beta + T) - log(alpha + T)
        third_term = log((1 - ((beta + T) / (beta + T + pred_weeks)) ** (s - 1))/(s - 1))
        return exp(first_term + second_term + third_term - likelihood)


    def pareto_total_conditional_prediction(self, pred_weeks):
        '''
        Make a single prediction for all train_df and return sum over a specific period of time
        returns a matrix of expected sales by customer and the total sales prediction
        '''
        self.train_df['pred'] = 0.
        self.train_df['actual'] = 0.
        pred_df = self.train_df[['customer_id', 'pred', 'actual']]
        self.pred_actual_matrix = pred_df.as_matrix()

        total_freq = 0
        for i in range(0, self.train_df.shape[0]):
            ID = self.train_df.iloc[i]['customer_id']
            x = self.train_df.iloc[i]['frequency']
            recency = self.train_df.iloc[i]['recency']
            T = self.train_df.iloc[i]['T']
            pred = self.pareto_conditional_expected_number_of_purchases_up_to_time(pred_weeks, x, recency, T)
            self.pred_actual_matrix[i][1] = pred
            total_freq += pred
        return self.pred_actual_matrix, total_freq


    def pareto_pred_and_actuals_matrix(self, test_df, pred_weeks):
        '''
        Adds column of actual values next to predicted values
        '''

        self.pred_actual_matrix, total_freq = self.pareto_total_conditional_prediction(pred_weeks)
        # add column of zeros to pred_matrix, fill in with actual values
        test_dict = split_test_df_by_pred_period(test_df, pred_weeks)

        for i in xrange(self.pred_actual_matrix.shape[0]):
            self.pred_actual_matrix[i][2] = test_dict.get(self.pred_actual_matrix[i][0], 0)
        return self.pred_actual_matrix


    def fit(self, train_df):
        res = self.maximize(train_df)
        if res.status != 0:
            raise Exception(res.message)
        self.r, self.alpha, self.a, self.b = res.x
        return self.r, self.alpha, self.a, self.b


    def params_(self):
        return 'r: {}, alpha: {}, a: {}, b: {}'.format(self.r, self.alpha, self.a, self.b)


    def create_ext_df(self, max_iter=150):
        '''
        - EXt = the expected number of transactions for a randomly-chosen individual in a time period of length t given r, alpha, a and b
        - returns df with the gaussian hypergeometric cost function for every day
        Iterates untill uj is close to 0 or maximum iterations is reached
        '''
        # self.r, self.alpha, self.a, self.b = 0.243, 4.414, 0.793, 2.426

        total_weeks = self.train_weeks + self.pred_weeks
        ext_df = pd.DataFrame(0, index=range(0, total_weeks), columns=['t', 'ext', '2F1', 'z'])

        ext_df['t'] = ext_df.apply(lambda x: x +1/7.).cumsum()

        ext_df['z'] = ext_df.t.apply(lambda x: x/ float(self.alpha + x))
        ext_df['2F1'] = ext_df.z.apply(lambda row: self.gaussian_hypergeometric(row, max_iter))

        ext_df['ext'] = ext_df.apply(lambda row: (self.a + self.b - 1)/ (self.a - 1) * (1 - (self.alpha/ (self.alpha + row['t'])) **self.r * row['2F1']), axis=1)
        return ext_df


    def gaussian_hypergeometric(self, row_z, max_iter):
        '''used for to calculate gaussian hypergeometric cost function, for every day, see lambda function 2F1'''
        gaus_cost = 1
        uj = 1
        for i in range(max_iter):
            u_new = uj * row_z * (self.r + i) * (self.b + i)/ ((self.a + self.b - 1 + i) * (i + 1))
            gaus_cost += u_new
            uj = u_new
        return gaus_cost


    def first_purchase_count_id(self):
        '''
        Counts the number of transactions per customer ID.
        '''
        self.train_df['first_purch'] = self.train_df.apply(lambda row: self.train_weeks - row['T'], axis = 1)
        first_purch_cnt = self.train_df.groupby('first_purch')['customer_id'].count().reset_index(name="cnt")
        first_purch_cnt.first_purch = first_purch_cnt.first_purch.round(2)

        first_purch_df = pd.DataFrame(0, index=range(0, self.first_purch_weeks * 7), columns=['first_purch'])
        first_purch_df['first_purch'] = first_purch_df.apply(lambda x: x +1/7.).cumsum().round(2)

        return first_purch_df.set_index('first_purch').join(first_purch_cnt.set_index('first_purch')).fillna(0).reset_index()


    def cummulative_repeated_sales(self, cum_weeks=10):
        '''
        Creates a forecast of repeat purchasing by calculating the expected number of weekly repeat transactions (cummulative)'''
        ext_df = self.create_ext_df()
        n_s = self.first_purchase_count_id()

        cum_rpt_sls = 0
        i_ext = cum_weeks * 7 - 2
        for i in xrange(cum_weeks * 7 - 1):
            cum_rpt_sls += ext_df['ext'][i_ext] * n_s['cnt'][i]
            i_ext -= 1
        return cum_rpt_sls


    def single_conditional_prediction(self, ID, x, recency, T, pred_weeks=''):
        '''
        For a randomly-chosen individual, computes the expected number of transactions in a time period of length t (pred_weeks) is

        Predicts a particular customer`s future purchasing, given information about his past behavior and the parameter estimates of the four models.
        '''
        self.max_iter = 150
        if pred_weeks == '':
            pred_weeks = self.pred_weeks

        a_id = self.r + x
        b_id = self.b + x
        c_id = self.a + self.b + x - 1
        z_id = pred_weeks/ (self.alpha + T + pred_weeks)

        gaus_cost = 1  # == 2F1 in paper
        uj = 1

        for i in xrange(1, self.max_iter):
            u_new = uj * (a_id + i - 1) * (b_id + i - 1)/ ((c_id + i - 1) * i) * z_id
            gaus_cost += u_new
            uj = u_new
        return (self.a + self.b + x - 1)/ (self.a - 1) * (1 - ((self.alpha + T) / (self.alpha + T + pred_weeks)) ** (self.r + x) * gaus_cost)/ (1 + (x > 0) * self.a / (self.b + x - 1) * ((self.alpha + T)/ (self.alpha + recency)) ** (self.r + x))


    def total_conditional_prediction(self, pred_weeks=''):
        '''
        Make a single prediction for all train_df and return sum over a specific period of time
        returns a matrix of expected sales by customer and the total sales prediction
        '''
        if pred_weeks == '':
            pred_weeks = self.pred_weeks

        self.train_df['pred'] = 0.
        self.train_df['actual'] = 0.
        pred_df = self.train_df[['customer_id', 'pred', 'actual']]
        self.pred_actual_matrix = pred_df.as_matrix()

        total_freq = 0
        for i in range(0, self.train_df.shape[0]):
            ID = self.train_df.iloc[i]['customer_id']
            x = self.train_df.iloc[i]['frequency']
            recency = self.train_df.iloc[i]['recency']
            T = self.train_df.iloc[i]['T']
            pred = self.single_conditional_prediction(ID, x, recency, T , pred_weeks)
            self.pred_actual_matrix[i][1] = pred
            total_freq += pred
        return self.pred_actual_matrix, total_freq

    def conditional_prediction_total_freq_only(self, pred_weeks):
        '''
        Similar than total_conditional_prediction but excludes making a matrix > faster
        Use this function for total_prediction_over_time
        '''

        total_freq = 0
        for i in range(0, self.train_df.shape[0]):
            ID = self.train_df.iloc[i]['customer_id']
            x = self.train_df.iloc[i]['frequency']
            recency = self.train_df.iloc[i]['recency']
            T = self.train_df.iloc[i]['T']

            pred = self.single_conditional_prediction(ID, x, recency, T , pred_weeks)
            total_freq += pred
        return total_freq


    def total_prediction_over_time(self, test_df, total_pred_weeks=39):
        '''
        Create dataframe from test, with a row for every day and create a cumsum for the num of transactions
        loop over every day and get the total_freq from total_conditional_prediction(), and append to dataframe
        '''
        # cut off transactions that are outside the prediction period
        test_df = test_df[test_df['test_weeks'] <= total_pred_weeks]

        # filter on same customer ids as used in the train_df. Group by date to get the num of transactions per date
        test_df = test_df[test_df.customer_id.isin(self.train_df['customer_id'].tolist())]

        # create empty df to fill in the results
        columns = ['act_cumsum', 'pred_x', 'pred_cumsum']
        index = np.linspace(1/float(7), total_pred_weeks, total_pred_weeks*7)
        pred_by_day_df = pd.DataFrame(index=index, columns=columns)
        pred_by_day_df = pred_by_day_df.fillna(0.) # with floats 0s rather than NaNs

        pred_by_day_df = pred_by_day_df.join(test_df.groupby('test_weeks')['cnt_trans'].sum())
        pred_by_day_df = pred_by_day_df.rename(columns={'cnt_trans': 'acrecency'})
        pred_by_day_df.acrecency = pred_by_day_df.acrecency.fillna(0)

        pred_by_day_df['act_cumsum'] = pred_by_day_df.acrecency.cumsum()

        pred_by_day_df = pred_by_day_df.reset_index()

        pred_by_day_df['pred_x'] = pred_by_day_df['index'].apply(lambda row: self.conditional_prediction_total_freq_only( pred_weeks=row))
        pred_by_day_df['pred_cumsum'] = pred_by_day_df.pred_x.cumsum()
        pred_by_day_df['perc_similar'] = pred_by_day_df['pred_cumsum'] / pred_by_day_df[ 'act_cumsum']

        return pred_by_day_df


    def pred_and_actuals_matrix(self, test_df, pred_weeks):
        '''
        Adds column of actual values next to predicted values
        '''
        if pred_weeks == '':
            pred_weeks = self.pred_weeks

        self.pred_actual_matrix, total_freq = self.total_conditional_prediction(pred_weeks)
        # add column of zeros to pred_matrix, fill in with actual values
        test_dict = split_test_df_by_pred_period(test_df, pred_weeks)

        for i in xrange(self.pred_actual_matrix.shape[0]):
            self.pred_actual_matrix[i][2] = test_dict.get(self.pred_actual_matrix[i][0], 0)
        return self.pred_actual_matrix


    def r2(self):
        y_true = self.pred_actual_matrix[:, 2]
        y_pred = self.pred_actual_matrix[:, 1]
        return r2_score(y_true, y_pred)


    def rmse(self):
        y_true = self.pred_actual_matrix[:, 2]
        y_pred = self.pred_actual_matrix[:, 1]
        return sqrt(mean_squared_error(y_true, y_pred))


    def mean_absolute_error(self):
        y_true = self.pred_actual_matrix[:, 2]
        y_pred = self.pred_actual_matrix[:, 1]
        return mean_absolute_error(y_true, y_pred)

    def rmsle(self):
        '''
        it basically acts as a % incorrect so it scales with parameter values
        '''
        y_true = self.pred_actual_matrix[:, 2].astype(float)
        y_pred = self.pred_actual_matrix[:, 1].astype(float)
        return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean())

if __name__ == '__main__':
    main()
#     '''Sample data consist of train_df who made their first transaction in week 0 to 12, we have info on their repeat purchasing behavior up to the week 39 and make predictions up to week 78'''

    # df = pd.read_csv('../data/test.csv', sep='\t')

    # df = pd.read_csv('../data/test/train_df_test.csv', sep='\t')
    # print df.head()
        # print 'r: {}, alpha: {}, a: {}, b: {}'.format(r, alpha, a, b)

    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.total_conditional_prediction(pred_weeks=10)

    # pred_matrix, total_freq = loaded_model.total_conditional_prediction(train_df)
    # r2_result = loaded_model.r2(test_df)
    # print bgnbd.total_conditional_prediction(pred_weeks=12, max_iter=2)
    # print lifetimes_fit(train_df)

    # check '381568065'
