import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sklearn
import shap 
import time
import sys # sys.exit() to debug

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# SECTION change directory #

os.chdir('directory')
 
# SECTION cross validate model parameters #

def find_combination(parameters):
    
    cross_validate_parameter_dict = {}

    parameters_keys = list(reversed(list(parameters.keys())))
    parameters_values = list(parameters.values())

    model_number = 0

    def nested_loop_recursion(depth,list_index,current=[]):

        nonlocal model_number # defines the variable model_number as nonlocal for use outside of nested loop

        value_list = parameters_values[list_index-1] # gets the value list from dictionary

        if depth == 0: # if iterated through all value lists then return current list
            
            single_parameter_dict = {}

            for count,key in enumerate(parameters_keys):

                single_parameter_dict[key] = current[count]

            model_number += 1

            model_name = f'model{model_number}'

            cross_validate_parameter_dict[model_name] = single_parameter_dict

        else:

            for value in value_list: # for every value in the value list

                nested_loop_recursion(depth-1,list_index-1,current+[value]) # we nest the loop by decreasing the list index until 0

                # essentially 
                # for value1 in parameter_list[2]: -> depth = 3
                #    for value2 in parameter_list[1]: -> depth = 2
                #       for value3 in parameter_list[0]: -> depth = 1
                #           print(value1,value2,value3)

    nested_loop_recursion(len(parameters_values),len(parameters_values))

    return cross_validate_parameter_dict

# SECTION standalone model #

class META_MODEL:

    def __init__(self,directory,dataframe_name,model_parameters=None,model_count=None,model=None,model_name=None):

        os.chdir(directory)
        self.df = pd.read_csv(dataframe_name)
        self.df = self.df.fillna(0)
        self.model_parameters = model_parameters
        self.model_count = model_count
        self.model = model
        self.model_name = model_name

        self.process()

    def process(self):

        self.feature_target()
        self.validation_split()
        self.imputation()
        self.scaling()
        self.k_fold()
        self.fit()

    def feature_target(self):
    
        self.df_feature = self.df.drop(['CAP'],axis='columns')
        self.df_target = self.df[['CAP']]

    def validation_split(self):

        self.df_train_feature, self.df_validation_feature, self.df_train_target, self.df_validation_target = train_test_split(self.df_feature,self.df_target,test_size=0.2,random_state=0)

        self.df_train_feature = self.df_train_feature.reset_index(drop=True)
        self.df_validation_feature = self.df_validation_feature.reset_index(drop=True)
        self.df_train_target = self.df_train_target.reset_index(drop=True)
        self.df_validation_target = self.df_validation_target.reset_index(drop=True)

    def imputation(self):

        # SECTION train #

        df_imputation_train = self.df_train_feature.drop(['CONC','CD'],axis='columns')
        df_imputation_train_extra = self.df_train_feature[['CONC','CD']]

        df_imputation_train_not_missing = df_imputation_train.iloc[df_imputation_train.index[df_imputation_train['DG'] != 0].tolist()]
        df_imputation_train_missing = df_imputation_train.iloc[df_imputation_train.index[df_imputation_train['DG'] == 0].tolist()]

        df_imputation_train_not_missing_feature = df_imputation_train_not_missing.drop(['DG'],axis='columns')
        df_imputation_train_not_missing_target = df_imputation_train_not_missing[['DG']]     

        df_imputation_train_missing_feature = df_imputation_train_missing.drop(['DG'],axis='columns')
        df_imputation_train_missing_target = df_imputation_train_missing[['DG']]  

        # SECTION validation #

        df_imputation_validation = self.df_validation_feature.drop(['CONC','CD'],axis='columns')
        df_imputation_validation_extra = self.df_validation_feature[['CONC','CD']]

        df_imputation_validation_not_missing = df_imputation_validation.iloc[df_imputation_validation.index[df_imputation_validation['DG'] != 0].tolist()]
        df_imputation_validation_missing = df_imputation_validation.iloc[df_imputation_validation.index[df_imputation_validation['DG'] == 0].tolist()]

        df_imputation_validation_not_missing_feature = df_imputation_validation_not_missing.drop(['DG'],axis='columns')
        df_imputation_validation_not_missing_target = df_imputation_validation_not_missing[['DG']]     

        df_imputation_validation_missing_feature = df_imputation_validation_missing.drop(['DG'],axis='columns')
        df_imputation_validation_missing_target = df_imputation_validation_missing[['DG']]  

        # SECTION train KNN model #

        KNN_imputor = KNeighborsRegressor(
            n_neighbors=3,
            weights='distance'
        )

        KNN_imputor.fit(df_imputation_train_not_missing_feature,df_imputation_train_not_missing_target)        

        # SECTION prediction #

        KNN_train_missing_prediction = KNN_imputor.predict(df_imputation_train_missing_feature)
        KNN_validation_missing_prediction = KNN_imputor.predict(df_imputation_validation_missing_feature)

        KNN_train_missing_prediction_df = pd.DataFrame(KNN_train_missing_prediction,columns=['DG']).set_index(df_imputation_train_missing_feature.index)
        KNN_validation_missing_prediction_df = pd.DataFrame(KNN_validation_missing_prediction,columns=['DG']).set_index(df_imputation_validation_missing_feature.index)

        df_fully_filled_train_missing = pd.concat([df_imputation_train_missing_feature,KNN_train_missing_prediction_df],axis='columns')
        df_fully_filled_validation_missing = pd.concat([df_imputation_validation_missing_feature,KNN_validation_missing_prediction_df],axis='columns')        

        combined_df_train = pd.concat([df_fully_filled_train_missing,df_imputation_train_not_missing],axis='rows')
        combined_df_validation = pd.concat([df_fully_filled_validation_missing,df_imputation_validation_not_missing],axis='rows')

        combined_df_train = pd.concat([combined_df_train,df_imputation_train_extra],axis='columns').sort_index()
        combined_df_validation = pd.concat([combined_df_validation,df_imputation_validation_extra],axis='columns').sort_index()

        self.df_train_feature = combined_df_train
        self.df_validation_feature = combined_df_validation
        self.df_train_target = self.df_train_target
        self.df_validation_target = self.df_validation_target

    def scaling(self):

        self.scaler_feature_dict = {}

        for column in self.df_train_feature:

            self.scaler_feature_dict[column] = {}
            self.scaler_feature_dict[column]['max'] = self.df_train_feature[column].max()
            self.scaler_feature_dict[column]['min'] = self.df_train_feature[column].min()    
            self.df_train_feature[column] = self.df_train_feature[column].apply(lambda x: (x-self.scaler_feature_dict[column]['min'])/(self.scaler_feature_dict[column]['max']-self.scaler_feature_dict[column]['min']))
            self.df_validation_feature[column] = self.df_validation_feature[column].apply(lambda x: (x-self.scaler_feature_dict[column]['min'])/(self.scaler_feature_dict[column]['max']-self.scaler_feature_dict[column]['min']))

        self.scaler_target_dict = {}

        for column in self.df_train_target:

            self.scaler_target_dict[column] = {}
            self.scaler_target_dict[column]['max'] = self.df_train_target[column].max()
            self.scaler_target_dict[column]['min'] = self.df_train_target[column].min()    
            self.df_train_target[column] = self.df_train_target[column].apply(lambda x: (x-self.scaler_target_dict[column]['min'])/(self.scaler_target_dict[column]['max']-self.scaler_target_dict[column]['min']))
            self.df_validation_target[column] = self.df_validation_target[column].apply(lambda x: (x-self.scaler_target_dict[column]['min'])/(self.scaler_target_dict[column]['max']-self.scaler_target_dict[column]['min']))

    def k_fold(self):

        self.number_of_folds = 10
        kfold_holder_for_training = KFold(n_splits=self.number_of_folds,random_state=None,shuffle=False)
        self.splits_for_training = kfold_holder_for_training.split(self.df_train_feature)

    def fit(self):

        self.train_start_time = time.time()

        self.model_dict = {}
        self.model_dict['XGB'] = {}
        self.model_dict['LGBM'] = {}
        self.model_dict['RF'] = {}
        self.model_dict['GB'] = {}
        self.model_dict['ADA'] = {}

        self.prediction_list = []
        self.META_hold_out_prediction_list = []

        for count_for_train, (train_index_for_validation,test_index_for_validation) in enumerate(self.splits_for_training):

            print(f'Fold {count_for_train}')

            # SECTION splits #

            kfold_train_feature_df = self.df_train_feature.iloc[train_index_for_validation]
            kfold_train_target_df = self.df_train_target.iloc[train_index_for_validation]

            kfold_test_feature_df = self.df_train_feature.iloc[test_index_for_validation]
            kfold_test_target_df = self.df_train_target.iloc[test_index_for_validation]

            # SECTION XGB #

            XGB_model = XGBRegressor(
                random_state=0,
                learning_rate=0.1,
                max_depth=None,
                n_estimators=500
            )

            XGB_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            XGB_model_prediction_for_meta_model = XGB_model.predict(kfold_test_feature_df)

            self.model_dict['XGB'][count_for_train] = {}
            self.model_dict['XGB'][count_for_train]['model'] = XGB_model
            self.model_dict['XGB'][count_for_train]['meta_feature'] = XGB_model_prediction_for_meta_model

            # SECTION LGBM #

            LGBM_model = LGBMRegressor(
                max_depth=10, 
                random_state=0,
                n_estimators=800,
                learning_rate=0.1
            )

            LGBM_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            LGBM_model_prediction_for_meta_model = LGBM_model.predict(kfold_test_feature_df)

            self.model_dict['LGBM'][count_for_train] = {}
            self.model_dict['LGBM'][count_for_train]['model'] = LGBM_model
            self.model_dict['LGBM'][count_for_train]['meta_feature'] = LGBM_model_prediction_for_meta_model

            # SECTION RF #

            RF_model = RandomForestRegressor(
                random_state=0,
                min_samples_leaf=1,
                min_samples_split=2,
                criterion='absolute_error',
                n_estimators=1000
            )

            RF_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            RF_model_prediction_for_meta_model = RF_model.predict(kfold_test_feature_df)

            self.model_dict['RF'][count_for_train] = {}
            self.model_dict['RF'][count_for_train]['model'] = RF_model
            self.model_dict['RF'][count_for_train]['meta_feature'] = RF_model_prediction_for_meta_model

            # SECTION GB #

            GB_model = GradientBoostingRegressor(
                random_state=0,
                min_samples_leaf=1,
                max_depth=3,
                n_estimators=800,
                learning_rate=0.3
            )

            GB_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            GB_model_prediction_for_meta_model = GB_model.predict(kfold_test_feature_df)

            self.model_dict['GB'][count_for_train] = {}
            self.model_dict['GB'][count_for_train]['model'] = GB_model
            self.model_dict['GB'][count_for_train]['meta_feature'] = GB_model_prediction_for_meta_model

            # SECTION ADA #

            ADA_model = AdaBoostRegressor(
                random_state=0,
                loss='square',
                n_estimators=800,
                learning_rate=1.5
            )

            ADA_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            ADA_model_prediction_for_meta_model = ADA_model.predict(kfold_test_feature_df)

            self.model_dict['ADA'][count_for_train] = {}
            self.model_dict['ADA'][count_for_train]['model'] = ADA_model
            self.model_dict['ADA'][count_for_train]['meta_feature'] = ADA_model_prediction_for_meta_model

            # SECTION stacking #
 
            XGB_meta_feature_df = pd.DataFrame(self.model_dict['XGB'][count_for_train]['meta_feature'],columns=['MODEL 1'])
            LGBM_meta_feature_df = pd.DataFrame(self.model_dict['LGBM'][count_for_train]['meta_feature'],columns=['MODEL 2'])
            RF_meta_feature_df = pd.DataFrame(self.model_dict['RF'][count_for_train]['meta_feature'],columns=['MODEL 3'])
            GB_meta_feature_df = pd.DataFrame(self.model_dict['GB'][count_for_train]['meta_feature'],columns=['MODEL 4'])
            ADA_meta_feature_df = pd.DataFrame(self.model_dict['ADA'][count_for_train]['meta_feature'],columns=['MODEL 5'])

            META_hold_out_prediction_df = pd.concat([XGB_meta_feature_df,LGBM_meta_feature_df,RF_meta_feature_df,GB_meta_feature_df,ADA_meta_feature_df],axis='columns')

            self.META_hold_out_prediction_list.append(META_hold_out_prediction_df)

        self.META_train_df = pd.concat(self.META_hold_out_prediction_list,axis='rows').reset_index(drop=True)

        # SECTION META model #

        if self.model is PolynomialRegressor:
    
            self.META_model = self.model.set_params(
                **self.model_parameters
            )

        else:

            self.META_model = self.model(
                **self.model_parameters
            )        

        self.META_model.fit(self.META_train_df,self.df_train_target.values.ravel())

        self.train_end_time = time.time()

        # SECTION refit XGB #

        self.refit_XGB_model = XGBRegressor(
                random_state=0,
                learning_rate=0.1,
                max_depth=None,
                n_estimators=500
            )
        self.refit_XGB_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

        # SECTION refit LGBM #

        self.refit_LGBM_model = LGBMRegressor(
                max_depth=10, 
                random_state=0,
                n_estimators=800,
                learning_rate=0.1
            )

        self.refit_LGBM_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

        # SECTION refit RF #

        self.refit_RF_model = RandomForestRegressor(
                random_state=0,
                min_samples_leaf=1,
                min_samples_split=2,
                criterion='absolute_error',
                n_estimators=1000
            )

        self.refit_RF_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

        # SECTION refit GB #

        self.refit_GB_model = GradientBoostingRegressor(
                random_state=0,
                min_samples_leaf=1,
                max_depth=3,
                n_estimators=800,
                learning_rate=0.3
            )

        self.refit_GB_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

        # SECTION refit ADA #

        self.refit_ADA_model = AdaBoostRegressor(
                random_state=0,
                loss='square',
                n_estimators=800,
                learning_rate=1.5
            )

        self.refit_ADA_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

    def predict(self):

        self.refit_XGB_prediction = pd.DataFrame(self.refit_XGB_model.predict(self.current_prediction_df[0]),columns=['MODEL 1'])
        self.refit_LGBM_prediction = pd.DataFrame(self.refit_LGBM_model.predict(self.current_prediction_df[0]),columns=['MODEL 2'])
        self.refit_RF_prediction = pd.DataFrame(self.refit_RF_model.predict(self.current_prediction_df[0]),columns=['MODEL 3'])
        self.refit_GB_prediction = pd.DataFrame(self.refit_GB_model.predict(self.current_prediction_df[0]),columns=['MODEL 4'])
        self.refit_ADA_prediction = pd.DataFrame(self.refit_ADA_model.predict(self.current_prediction_df[0]),columns=['MODEL 5'])

        self.META_prediction_df = pd.concat([self.refit_XGB_prediction,self.refit_LGBM_prediction,self.refit_RF_prediction,self.refit_GB_prediction,self.refit_ADA_prediction],axis='columns')

        self.META_prediction = pd.DataFrame(self.META_model.predict(self.META_prediction_df),columns=['AVG PREDICTION'])

    def average_prediction(self):

        # SECTION get index #

        self.result_df = pd.DataFrame()

        for column in self.current_prediction_df[0]:

            self.result_df[column] = self.current_prediction_df[0][column].apply(lambda x: (x * (self.scaler_feature_dict[column]['max'] - self.scaler_feature_dict[column]['min'])) + self.scaler_feature_dict[column]['min'])

        for column in self.current_prediction_df[1]:
            
            self.result_df[column] = self.current_prediction_df[1][column].apply(lambda x: (x * (self.scaler_target_dict[column]['max'] - self.scaler_target_dict[column]['min'])) + self.scaler_target_dict[column]['min'])

        self.result_df['PREDICTION'] = (self.META_prediction['AVG PREDICTION'] * (self.scaler_target_dict['CAP']['max'] - self.scaler_target_dict['CAP']['min'])) + self.scaler_target_dict['CAP']['min']
        self.result_df['DIFFERENCE'] = abs(self.result_df['CAP'] - self.result_df['PREDICTION'])

    def get_error_df(self):

        self.error_df = pd.DataFrame(columns=['ERROR'])
        self.error_df['ERROR'] = abs(self.result_df['CAP'] - self.result_df['PREDICTION'])

    def plot(self):

        # SECTION plot predictions #

        performance_fig = plt.figure(figsize=(20,11.25))
        performance_fig.suptitle(f'{self.current_prediction_df[2]} {self.model_name} {self.model_count}')

        mean_error = self.error_df['ERROR'].mean()
        min_error = self.error_df['ERROR'].min()
        max_error = self.error_df['ERROR'].max()
        median_error = self.error_df['ERROR'].median()

        ax1 = plt.subplot2grid((2,2),(0,0),fig=performance_fig) # 2 x 2 graph with location at 0,0
        ax1.scatter(self.result_df['DIFFERENCE'].index.values,self.result_df['PREDICTION'],color='red',marker='.',label=f'prediction error mean {mean_error:.2f} max {max_error:.2f} min {min_error:.2f} median {median_error:.2f}')
        ax1.scatter(self.result_df['DIFFERENCE'].index.values,self.result_df['CAP'],color='blue',marker='.',label='real')
        ax1.set_xlabel('Data point')
        ax1.set_ylabel('Capacitance')

        for count in range(len(self.result_df['DIFFERENCE'])):

            ax1.vlines(count,ymin=self.result_df['PREDICTION'].iloc[count],ymax=self.result_df['CAP'].iloc[count],color='black')

        ax1.legend()

        ax2 = plt.subplot2grid((2,2),(0,1),fig=performance_fig) # 2 x 2 graph with location at 0,1
        ax2.plot(self.result_df['PREDICTION'],color='red',marker='.',label=f'prediction error mean {mean_error:.2f} max {max_error:.2f} min {min_error:.2f} median {median_error:.2f}')
        ax2.plot(self.result_df['CAP'],color='blue',marker='.',label='real')
        ax2.set_xlabel('Data point')
        ax2.set_ylabel('Capacitance')

        ax2.legend()

        ax3 = plt.subplot2grid((2,2),(1,0),colspan=2,fig=performance_fig) # 2 x 2 graph with location at 1,0 that spans 2 slots
        ax3.scatter(self.result_df['CAP'],self.result_df['PREDICTION'],color='red',marker='x',label=f'real(x) vs prediction(y)')
        real_max = self.result_df['CAP'].max()
        predict_max = self.result_df['PREDICTION'].max()
        line_list = [0,real_max if real_max > predict_max else predict_max]
        ax3.plot(line_list,line_list,color='black')
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Prediction')

        performance_fig.supxlabel(f'Execution time {(self.train_end_time - self.train_start_time) / 60:.2f} minutes')
        
        performance_fig.savefig(f'{self.current_prediction_df[2]}{self.model_name}{self.model_count}.jpg',dpi=performance_fig.dpi)
        plt.close()

    def train_and_validation_testing(self):

        self.train_and_validation_testing_set = [[self.df_train_feature,self.df_train_target,'train'],[self.df_validation_feature,self.df_validation_target,'validation']]
        self.return_list = []

        for dataframe in self.train_and_validation_testing_set:
    
            self.current_prediction_df = dataframe
            self.predict()
            self.average_prediction()
            self.get_error_df()
            # self.plot()

            model_result_df = pd.DataFrame([[self.model_name,self.model_parameters,self.error_df['ERROR'].mean(),self.error_df['ERROR'].min(),self.error_df['ERROR'].max(),self.error_df['ERROR'].median(),self.train_end_time-self.train_start_time]],columns=['MODEL','PARAMETERS','MEAN','MIN','MAX','MEDIAN','TIME'])
            self.return_list.append(model_result_df)

        return self.return_list 

# SECTION model searching #

LGBM_parameters = {'random_state':[0],'n_jobs':[-1]}
XGB_parameters = {'random_state':[0],'n_jobs':[-1]}
POLY_parameters = {'poly__degree':[2,3,4]}
NN_parameters = {'hidden_layer_sizes':[(8,8),(32,64,256,256),(32,32,32)],'activation':['relu'],'solver':['adam'],'learning_rate_init':[0.001],'random_state':[0]}
ELAS_parameters = {'random_state':[0]}
LASS_parameters= {'alpha':[1]}
RIDGE_parameters = {'alpha':[1]}
RF_parameters = {'random_state':[0],'n_jobs':[-1]}
SVM_parameters = {'kernel':['linear','poly','rbf']}
KNN_parameters = {'weights':['uniform','distance'],'n_jobs':[-1]}
GB_parameters = {'random_state':[0]}
ADA_parameters = {'random_state':[0]}
DEC_parameters = {'random_state':[0]}

PolynomialRegressor = Pipeline([
  ('poly',PolynomialFeatures(degree=2)),
  ('linear',LinearRegression(fit_intercept=False))
])

parameter_list = [
    ['LGBM',LGBM_parameters,LGBMRegressor],
    ['XGB',XGB_parameters,XGBRegressor],
    ['POLY',POLY_parameters,PolynomialRegressor],
    ['NN',NN_parameters,MLPRegressor],
    ['ELAS',ELAS_parameters,ElasticNet],
    ['LASS',LASS_parameters,Lasso],
    ['RIDGE',RIDGE_parameters,Ridge],
    ['RF',RF_parameters,RandomForestRegressor],
    ['SVM',SVM_parameters,SVR],
    ['KNN',KNN_parameters,KNeighborsRegressor],
    ['GB',GB_parameters,GradientBoostingRegressor],
    ['ADA',ADA_parameters,AdaBoostRegressor],
    ['DEC',DEC_parameters,DecisionTreeRegressor],
]

model_train_performance_df = pd.DataFrame(columns=['MODEL','PARAMETERS','MEAN','MAX','MIN','MEDIAN','TIME'])
model_validation_performance_df = pd.DataFrame(columns=['MODEL','PARAMETERS','MEAN','MAX','MIN','MEDIAN','TIME'])

for parameter in parameter_list:

    model_name = parameter[0]
    parameters = parameter[1]
    model = parameter[2]

    model_testing_dict = find_combination(parameters)

    for model_count,test_parameters in enumerate(model_testing_dict.values()):
        
        meta_model = META_MODEL(
            directory='directory',
            dataframe_name='data.csv',     
            model_parameters=test_parameters,
            model_count=model_count,
            model=model,
            model_name=model_name
        )

        model_result_df = meta_model.train_and_validation_testing()

        model_train_performance_df = pd.concat([model_train_performance_df,model_result_df[0]],axis='rows')
        model_validation_performance_df = pd.concat([model_validation_performance_df,model_result_df[1]],axis='rows')

        print(model_train_performance_df)
        print(model_validation_performance_df)

model_train_performance_df.to_csv('meta_train.csv')
model_validation_performance_df.to_csv('meta_validation.csv')