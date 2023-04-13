import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

class ASSEMBLED_STACKING_MODEL:

    def __init__(self,directory,dataframe_name,shap=False,optimization=False,model_parameters=None,model_count=None,optimized_model=None,validation_split_random_state=0):

        os.chdir(directory)
        self.df = pd.read_csv(dataframe_name)
        self.df = self.df.fillna(0)
        self.shap = shap
        self.optimization = optimization
        self.model_parameters = model_parameters
        self.model_count = model_count
        self.optimized_model = optimized_model
        self.shap_start = 0
        self.shap_end = 30
        self.validation_split_random_state = validation_split_random_state

        self.process()

    def process(self):

        self.feature_target()
        self.validation_split()
        self.imputation()
        self.scaling()
        self.k_fold()
        self.fit()

        for dataframe in self.train_and_validation_testing:

            self.current_prediction_df = dataframe
            self.predict()
            self.average_prediction()
            self.plot()

            plt.close()

    def feature_target(self):
        
        self.df_feature = self.df.drop(['CAP'],axis='columns')
        self.df_target = self.df[['CAP']]

    def validation_split(self):

        self.df_train_feature, self.df_validation_feature, self.df_train_target, self.df_validation_target = train_test_split(self.df_feature,self.df_target,test_size=0.2,random_state=self.validation_split_random_state)

        self.df_train_feature = self.df_train_feature.reset_index(drop=True)
        self.df_validation_feature = self.df_validation_feature.reset_index(drop=True)
        self.df_train_target = self.df_train_target.reset_index(drop=True)
        self.df_validation_target = self.df_validation_target.reset_index(drop=True)

    def imputation(self):

        # SECTION train #

        df_imputation_train = self.df_train_feature.drop(['CONC','CD','RDXPO','DISS'],axis='columns')
        df_imputation_train_extra = self.df_train_feature[['CONC','CD','RDXPO','DISS']]

        df_imputation_train_not_missing = df_imputation_train.iloc[df_imputation_train.index[df_imputation_train['DG'] != 0].tolist()]
        df_imputation_train_missing = df_imputation_train.iloc[df_imputation_train.index[df_imputation_train['DG'] == 0].tolist()]

        df_imputation_train_not_missing_feature = df_imputation_train_not_missing.drop(['DG'],axis='columns')
        df_imputation_train_not_missing_target = df_imputation_train_not_missing[['DG']]     

        df_imputation_train_missing_feature = df_imputation_train_missing.drop(['DG'],axis='columns')
        df_imputation_train_missing_target = df_imputation_train_missing[['DG']]  

        # SECTION validation #

        df_imputation_validation = self.df_validation_feature.drop(['CONC','CD','RDXPO','DISS'],axis='columns')
        df_imputation_validation_extra = self.df_validation_feature[['CONC','CD','RDXPO','DISS']]

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

        from sklearn.model_selection import KFold

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
                n_estimators=300
            )

            XGB_model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
            XGB_model_prediction_for_meta_model = XGB_model.predict(kfold_test_feature_df)

            self.model_dict['XGB'][count_for_train] = {}
            self.model_dict['XGB'][count_for_train]['model'] = XGB_model
            self.model_dict['XGB'][count_for_train]['meta_feature'] = XGB_model_prediction_for_meta_model

            # SECTION LGBM #

            LGBM_model = LGBMRegressor(
                max_depth=0, 
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
                min_samples_leaf=2,
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

        META_model1 = Ridge(
            alpha=1
        )

        META_model1.fit(self.META_train_df,self.df_train_target.values.ravel())

        META_model2 = MLPRegressor(
            random_state=0,
            learning_rate_init=0.001,
            solver='adam',
            activation='relu',
            hidden_layer_sizes=(32,64,256,256)
        )

        META_model2.fit(self.META_train_df,self.df_train_target.values.ravel())

        META_model3 = RandomForestRegressor(
            n_jobs=-1,
            random_state=0
        )

        META_model3.fit(self.META_train_df,self.df_train_target.values.ravel())

        self.META_model1 = META_model1
        self.META_model2 = META_model2
        self.META_model3 = META_model3

        self.train_end_time = time.time()

        # SECTION refit base models

        # SECTION refit XGB #

        self.refit_XGB_model = XGBRegressor(
                random_state=0,
                learning_rate=0.1,
                max_depth=None,
                n_estimators=300
        )

        self.refit_XGB_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

        # SECTION refit LGBM #

        self.refit_LGBM_model = LGBMRegressor(
                max_depth=0, 
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
                min_samples_leaf=2,
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

        self.train_and_validation_testing = [[self.df_train_feature,self.df_train_target,'train'],[self.df_validation_feature,self.df_validation_target,'validation']]

    def predict(self):

        self.refit_XGB_prediction = pd.DataFrame(self.refit_XGB_model.predict(self.current_prediction_df[0]),columns=['MODEL 1'])
        self.refit_LGBM_prediction = pd.DataFrame(self.refit_LGBM_model.predict(self.current_prediction_df[0]),columns=['MODEL 2'])
        self.refit_RF_prediction = pd.DataFrame(self.refit_RF_model.predict(self.current_prediction_df[0]),columns=['MODEL 3'])
        self.refit_GB_prediction = pd.DataFrame(self.refit_GB_model.predict(self.current_prediction_df[0]),columns=['MODEL 4'])
        self.refit_ADA_prediction = pd.DataFrame(self.refit_ADA_model.predict(self.current_prediction_df[0]),columns=['MODEL 5'])

        self.META_prediction_df = pd.concat([self.refit_XGB_prediction,self.refit_LGBM_prediction,self.refit_RF_prediction,self.refit_GB_prediction,self.refit_ADA_prediction],axis='columns')

        self.META_model1_prediction = pd.DataFrame(self.META_model1.predict(self.META_prediction_df),columns=['AVG PREDICTION'])
        self.META_model2_prediction = pd.DataFrame(self.META_model2.predict(self.META_prediction_df),columns=['AVG PREDICTION'])
        self.META_model3_prediction = pd.DataFrame(self.META_model3.predict(self.META_prediction_df),columns=['AVG PREDICTION'])

        self.META_prediction = (self.META_model1_prediction + self.META_model2_prediction + self.META_model3_prediction) / 3

    def average_prediction(self):

        # SECTION get index #

        self.result_df = pd.DataFrame()

        for column in self.current_prediction_df[0]:

            self.result_df[column] = self.current_prediction_df[0][column].apply(lambda x: (x * (self.scaler_feature_dict[column]['max'] - self.scaler_feature_dict[column]['min'])) + self.scaler_feature_dict[column]['min'])

        for column in self.current_prediction_df[1]:
            
            self.result_df[column] = self.current_prediction_df[1][column].apply(lambda x: (x * (self.scaler_target_dict[column]['max'] - self.scaler_target_dict[column]['min'])) + self.scaler_target_dict[column]['min'])

        self.result_df['PREDICTION'] = (self.META_prediction['AVG PREDICTION'] * (self.scaler_target_dict['CAP']['max'] - self.scaler_target_dict['CAP']['min'])) + self.scaler_target_dict['CAP']['min']
        self.result_df['DIFFERENCE'] = abs(self.result_df['CAP'] - self.result_df['PREDICTION'])

    def plot(self):

        # SECTION plot predictions #

        performance_fig = plt.figure(figsize=(20,11.25))
        performance_fig.suptitle(f'{self.current_prediction_df[2]} META')

        error_df = pd.DataFrame(columns=['ERROR'])
        error_df['ERROR'] = abs(self.result_df['CAP'] - self.result_df['PREDICTION'])
        mean_error = error_df['ERROR'].mean()
        min_error = error_df['ERROR'].min()
        max_error = error_df['ERROR'].max()
        median_error = error_df['ERROR'].median()

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
        performance_fig.savefig(f'randomstate{self.validation_split_random_state}final{self.current_prediction_df[2]}METAmodel.jpg',dpi=performance_fig.dpi)

for random_state in range(10):

    stacking_model = ASSEMBLED_STACKING_MODEL(
        directory='directory',
        dataframe_name='data.csv',
        validation_split_random_state=random_state
    )

print('Done')
