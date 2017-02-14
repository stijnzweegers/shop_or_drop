import pickle
import pandas as pd
import numpy as np


from bgnbd import BgNbd
from convert_df import txt_to_df, df_transform, train_test_split, split_by_freq
from convert_df import undersampling, oversampling
from churn_rate import avg_churn_rate

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


### training data parameters
data_source = '../data/atk_transaction.txt'
first_purchasing_weeks = 39
train_weeks = 100
# pred_weeks = 20
site_id = 31791
frequency_split = True
frequency_split_num = 4
undersample = False
oversample = False
sample_threshold = 0.15
split_major = 4

### model parameters
model = 'bgnbd'
penalizer_coef = 0.0
# pred_weeks_score = pred_weeks  #default is equal to  training data pred_weeks
max_iter = 150

### pickle names
pickle_train_df_name = 'train_df'
pickle_test_df_name = 'test_df'
pickle_model_name = 'bgnbd_one_comp'



def load_df(split_freq=frequency_split):
    '''
    Run this function when you changed the global training data parameters
    '''
    df = txt_to_df(data_source)
    print '------df loading done-----'
    transformed_df = df_transform(df, site_id=site_id)
    print '----transformed df done----'
    train_df, test_df, first_purch_weeks, train_end_date = train_test_split(transformed_df, first_purch_weeks=first_purchasing_weeks, train_end_week=train_weeks)
    print '---train test split done---'
    test_df.to_pickle(pickle_test_df_name+'.pickle')
    if split_freq == False:
        ### pickle transformed dataframes
        train_df.to_pickle(pickle_train_df_name+'.pickle')
        print '---pickle files done---'
        return train_df, test_df, first_purch_weeks, train_end_date
    if split_freq == True:
        train_df_high_freq, train_df_low_freq = split_by_freq(train_df, frequency_split_num)
        train_df_high_freq.to_pickle(pickle_train_df_name+'_high_freq.pickle')
        train_df_low_freq.to_pickle(pickle_train_df_name+'_low_freq.pickle')
        print '---pickle files done---'
        return train_df_high_freq, train_df_low_freq, test_df, first_purch_weeks, train_end_date


def load_pickled_df(split_freq=frequency_split):
    '''
    Always run this function, your train and test df are always pickled
    '''
    if split_freq == False:
        train_df = pd.read_pickle(pickle_train_df_name+'.pickle')
        test_df = pd.read_pickle(pickle_test_df_name+'.pickle')
        return train_df, test_df
    if split_freq == True:
        train_df_high_freq = pd.read_pickle(pickle_train_df_name+'_high_freq.pickle')
        train_df_low_freq = pd.read_pickle(pickle_train_df_name+'_low_freq.pickle')
        test_df = pd.read_pickle(pickle_test_df_name+'.pickle')
        return train_df_high_freq, train_df_low_freq, test_df


def train_model_bgnbd(split_freq=frequency_split):
    '''
    Run this function when using a new training data set
    '''
    if split_freq == False:
        if undersample == True:
            undersampled_train_df = undersampling(train_df, sample_threshold, split=split_major)
            print '--training--len undersampled train data: {} --'.format(len(undersampled_train_df))
            bgnbd = BgNbd(undersampled_train_df, first_purch_weeks=first_purchasing_weeks, train_weeks=train_weeks, max_iter=max_iter)
            if model == 'bgnbd':
                bgnbd_fit_params = bgnbd.lifetimes_fit(undersampled_train_df, penalizer_coef=penalizer_coef)
            if model == 'modified_bgnbd':
                bgnbd_fit_params = bgnbd.modified_fit(undersampled_train_df, penalizer_coef=penalizer_coef)
            print 'bgnbd_fit_params:{}'.format(bgnbd_fit_params)
            pickle.dump(model, open(pickle_model_name+'.sav', 'wb'))
        if oversample == True:
            oversampled_train_df = oversampling(train_df, sample_threshold, split=split_major)
            print '--training--len oversampled train data: {} --'.format(len(oversampled_train_df))
            bgnbd = BgNbd(oversampled_train_df, first_purch_weeks=first_purchasing_weeks, train_weeks=train_weeks, max_iter=max_iter)
            if model == 'bgnbd':
                bgnbd_fit_params = bgnbd.lifetimes_fit(oversampled_train_df, penalizer_coef=penalizer_coef)
            if model == 'modified_bgnbd':
                bgnbd_fit_params = bgnbd.modified_fit(oversampled_train_df, penalizer_coef=penalizer_coef)

            print 'bgnbd_fit_params:{}'.format(bgnbd_fit_params)
            pickle.dump(bgnbd, open(pickle_model_name+'.sav', 'wb'))
        if oversample == False and undersample == False:
            print '--------training--len train data: {} -----'.format(len(train_df))
            bgnbd = BgNbd(train_df, first_purch_weeks=first_purchasing_weeks, train_weeks=train_weeks, max_iter=max_iter)
            if model == 'bgnbd':
                bgnbd_fit_params = bgnbd.lifetimes_fit(train_df, penalizer_coef=penalizer_coef)
            if model == 'modified_bgnbd':
                bgnbd_fit_params = bgnbd.modified_fit(train_df, penalizer_coef=penalizer_coef)

            # pareto
            if model == 'pareto':
                pareto_fit_params = bgnbd.pareto_fit(train_df)

            print model+'_fit_params:{}'.format(pareto_fit_params)
            pickle.dump(bgnbd, open(pickle_model_name+'.sav', 'wb'))

    if split_freq == True:
        bgnbd_high_freq = BgNbd(train_df_high_freq, first_purch_weeks=first_purchasing_weeks, train_weeks=train_weeks, max_iter=max_iter)
        if model == 'bgnbd':
            bgnbd_fit_params_high_freq = bgnbd_high_freq.lifetimes_fit(train_df_high_freq, penalizer_coef=penalizer_coef)
        if model == 'modified_bgnbd':
            bgnbd_fit_params_high_freq = bgnbd_high_freq.modified_fit(train_df_high_freq, penalizer_coef=penalizer_coef)
        print 'bgnbd fit params high freq:{}'.format(bgnbd_fit_params_high_freq)
        pickle.dump(bgnbd_high_freq, open(pickle_model_name+'high_freq.sav', 'wb'))

        bgnbd_low_freq = BgNbd(train_df_low_freq, first_purch_weeks=first_purchasing_weeks, train_weeks=train_weeks, max_iter=max_iter)
        if model == 'bgnbd':
            bgnbd_fit_params_low_freq = bgnbd_low_freq.lifetimes_fit(train_df_low_freq, penalizer_coef=penalizer_coef)
        if model == 'modified_bgnbd':
            bgnbd_fit_params_low_freq = bgnbd_low_freq.modified_fit(train_df_low_freq, penalizer_coef=penalizer_coef)
        print 'bgnbd fit params low freq:{}'.format(bgnbd_fit_params_low_freq)
        pickle.dump(bgnbd_low_freq, open(pickle_model_name+'low_freq.sav', 'wb'))


def run_pickled_model(pred_weeks, split_freq=frequency_split):
    '''
    Always run this function, your model is always pickled
    '''
    if model == 'pareto':
        loaded_model = pickle.load(open(pickle_model_name+'.sav', 'rb'))
        train_df, test_df = load_pickled_df()
        pareto_pred_and_actuals_matrix = loaded_model.pareto_pred_and_actuals_matrix(test_df, pred_weeks=39)
        return pareto_pred_and_actuals_matrix

    if split_freq == False:
        loaded_model = pickle.load(open(pickle_model_name+'.sav', 'rb'))
        print loaded_model.params_()
        train_df, test_df = load_pickled_df()
        pred_actual_matrix = loaded_model.pred_and_actuals_matrix(test_df, pred_weeks)
        print '--predicted actual matrix (done)--'
        print '------------R2--------------'
        print loaded_model.r2()
        print '------------RMSE--------------'
        print loaded_model.rmse()
        print '----Mean Absolute Error----'
        print loaded_model.mean_absolute_error()
        print '----RMSLE----'
        print loaded_model.rmsle()
        # return pred_actual_matrix
        y_true = pred_actual_matrix[:, 2]
        y_pred = pred_actual_matrix[:, 1]
        return y_true.sum(), y_pred.sum()

    if split_freq == True:
        print 'pred_weeks: {}'.format(pred_weeks)
        loaded_model_high_freq = pickle.load(open(pickle_model_name+'high_freq.sav', 'rb'))
        print 'high freq params: {}'.format(loaded_model_high_freq.params_())
        loaded_model_low_freq = pickle.load(open(pickle_model_name+'low_freq.sav', 'rb'))
        print 'low freq params: {}'.format(loaded_model_low_freq.params_())
        train_df_high_freq, train_df_low_freq, test_df = load_pickled_df()

        pred_actual_matrix_high_freq = loaded_model_high_freq.pred_and_actuals_matrix(test_df, pred_weeks)
        pred_actual_matrix_low_freq = loaded_model_low_freq.pred_and_actuals_matrix(test_df, pred_weeks)
        print '--predicted actual matrices (done)--'
        print '---separate scores---'
        print 'R2 high freq: {}'.format(loaded_model_high_freq.r2())
        print 'R2 low freq: {}'.format(loaded_model_low_freq.r2())
        print 'RMSE high freq: {}'.format(loaded_model_high_freq.rmse())
        print 'RMSE low freq: {}'.format(loaded_model_low_freq.rmse())
        print 'Mean Abs Error high freq: {}'.format(loaded_model_high_freq.mean_absolute_error())
        print 'Mean Abs Error low freq: {}'.format(loaded_model_low_freq.mean_absolute_error())
        print 'RMSLE high freq: {}'.format(loaded_model_high_freq.rmsle())
        print 'RMSLE low freq: {}'.format(loaded_model_low_freq.rmsle())
        print '----combined scores----'
        combined_pred_actual_matrix = np.vstack([pred_actual_matrix_high_freq, pred_actual_matrix_low_freq])
        y_true = combined_pred_actual_matrix[:, 2].astype(float)
        y_pred = combined_pred_actual_matrix[:, 1].astype(float)
        print 'R2 combined: {}'.format(r2_score(y_true, y_pred))
        print 'RMSE combined: {}'.format(sqrt(mean_squared_error(y_true, y_pred)))
        print 'Mean Abs Error combined: {}'.format(mean_absolute_error(y_true, y_pred))
        print 'RMSLE combined: {}'.format(np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean()))

        return y_true.sum(), y_pred.sum()

        # return pred_actual_matrix_high_freq, pred_actual_matrix_low_freq


def make_prediction_over_time(total_pred_weeks=30):
    if frequency_split == False:
        loaded_model = pickle.load(open(pickle_model_name+'.sav', 'rb'))
        print loaded_model.params_()
        train_df, test_df = load_pickled_df()
        pred_by_day_df = loaded_model.total_prediction_over_time(test_df, total_pred_weeks)
        # return loaded_model.conditional_prediction_total_freq_only(10)
        return pred_by_day_df


if __name__ == '__main__':
    print '---run site id {}---'.format(site_id)
    print 'Params: model: {}, first_purchasing_weeks: {}, train_weeks: {}, frequency_split: {}, frequency_split_num: {}, penalizer_coef: {}'.format(model, first_purchasing_weeks, train_weeks, frequency_split, frequency_split_num, penalizer_coef)

    if model == 'pareto':
        # train_df, test_df, first_purch_weeks, train_end_date = load_df()
        # train_df, test_df = load_pickled_df()
        # train_model_bgnbd()
        pareto_pred_actual_matrix = run_pickled_model(pred_weeks=39)
    else:
        if frequency_split == False:
            '''
            Run when changes in data and/or model settings, no split
            '''
            train_df, test_df, first_purch_weeks, train_end_date = load_df()
            train_df, test_df = load_pickled_df()
            train_model_bgnbd()
            '''
            Run when no changes in data, no split
            '''
            # pred_actual_matrix = run_pickled_model(pred_weeks=39)
            # pred_by_day_df = make_prediction_over_time()
            weekly_predictions = np.zeros((40, 3))
            for week in np.arange(0, 40):
                y_true_sum , y_pred_sum = run_pickled_model(pred_weeks=week)
                weekly_predictions[week][0] = week
                weekly_predictions[week][1] = y_true_sum
                weekly_predictions[week][2] = y_pred_sum
                print week, y_true_sum, y_pred_sum
            weekly_predictions

        if frequency_split == True:
            '''
            Run when changes in data and/ or model settings and data is splitted in high/ low freq
            '''

            train_df_high_freq, train_df_low_freq, test_df, first_purch_weeks, train_end_date = load_df()
            train_df_high_freq, train_df_low_freq, test_df = load_pickled_df()
            train_model_bgnbd()
            '''
            Run when no changes in data, no split
            '''
            pred_actual_matrix_high_freq, pred_actual_matrix_low_freq = run_pickled_model(pred_weeks=39)
            # weekly_predictions = np.zeros((40, 3))
            # for week in np.arange(0, 40):
            #     y_true_sum , y_pred_sum = run_pickled_model(pred_weeks=week)
            #     weekly_predictions[week][0] = week
            #     weekly_predictions[week][1] = y_true_sum
            #     weekly_predictions[week][2] = y_pred_sum
            #     print week, y_true_sum, y_pred_sum
            # weekly_predictions
