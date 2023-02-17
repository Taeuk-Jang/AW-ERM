import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
import os, urllib
from torch.utils.data import Dataset

import json, os, string, random, time, pickle, gc, pdb
from PIL import Image
from PIL import ImageFilter
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from sklearn.preprocessing import OneHotEncoder

class CelebALoader(data.Dataset):
    def __init__(self, args, annotation_dir, split = 'train', transform = None):
        print("CelebA dataloader")

        self.split = split
        self.annotation_dir = annotation_dir
        self.transform = transform

        print("loading %s annotations.........." % self.split)
        self.df = pd.read_csv(open(os.path.join(annotation_dir, split+".csv"), 'rb'))

        self.labels = np.eye(2)[self.df['Attractive'].values]
        self.gender = np.eye(2)[self.df['Male'].values]
        
        print("dataset size: %d" % len(self.labels))

        self.image_dir = self.df['Path'].values
        self.image_ids = [i.split('/')[-1] for i in self.df['Path'].values]

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender[:, 1])[0]), len(np.nonzero(self.gender[:, 0])[0])))

    def __getitem__(self, index):
        image_dir = self.image_dir[index]
        
        image_ids = self.image_ids[index]
        
        img_ = Image.open(image_dir).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.labels[index]), \
                torch.LongTensor(self.gender[index]), [self.image_ids[index]]
    def __len__(self):
        return len(self.gender)

    
class tabular_dataset(Dataset):
    def __init__(self, X, Y, A, std = None, noise = True, M=2):
        self.X_raw = X
        self.Y = Y
        self.A = A
        self.noise = noise
        self.std = std
        self.M = M
        
#         if noise:
#             self.X = []
#             self.X.append((self.X_raw + np.random.normal(0,std,self.X_raw.shape), 0, 1))
#             self.X.append((self.X_raw + np.random.normal(0,std,self.X_raw.shape), 0, 1))
#         else:
#             self.X = self.X_raw
            
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        
        if self.noise:
            return [self.X_raw[idx] + np.random.normal(0, self.std) for i in range(self.M)], self.Y[idx], self.A[idx]
        else:
            return self.X_raw[idx], self.Y[idx], self.A[idx]

def get_dataset(name, save=False, corr_sens=False, seed=42, verbose=False):
    """
    Retrieve dataset and all relevant information
    :param name: name of the dataset
    :param save: if set to True, save the dataset as a pickle file. Defaults to False
    :return: Preprocessed dataset and relevant information
    """
    def get_numpy(df):
        new_df = df.copy()
        cat_columns = new_df.select_dtypes(['category']).columns
        new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
        return new_df.values

    if name == 'adult':
        # Load data
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', \
                         'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
        df = pd.read_csv('/data/adult/adult.data', names=feature_names)
        
        if verbose:
            print('Raw Dataset loaded.')
        num_train = df.shape[0]
        pos_class_label = ' >50K'
        neg_class_label = ' <=50K'
        y = np.zeros(num_train)
        y[df.iloc[:,-1].values == pos_class_label] = 1
#         df = df.drop(['fnlwgt', 'education-num'], axis=1)
        df = df.drop(['fnlwgt'], axis=1)
        num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
#         cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
        feature_names = num_var_names + cat_var_names
        df = df[feature_names]
        df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship'], prefix_sep='=')
        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = df.columns.get_loc('race')
        sex_idx = df.columns.get_loc('sex')
#         print( df.columns.get_loc('sex'))
        sens_idc = [race_idx, sex_idx]

        race_cats = df.iloc[:, race_idx].cat.categories
        sex_cats = df.iloc[:, sex_idx].cat.categories
        
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')

    elif name == 'bank':
        # Load data
        
        df = pd.read_csv('/data/bank/bank-additional-full.csv', sep = ';', na_values=['unknown'])

        df['age'] = df['age'].apply(lambda x: x >= 25)
        df = df[np.array(df.default == 'no') + np.array(df.default == 'yes')]

        #         num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_var_names = ['job', 'marital', 'education', 'default',
                         'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome']
        #         feature_names = num_var_names + cat_var_names
        #         df = df[feature_names]

#         df = df.drop(['default'], axis=1)
        df = pd.get_dummies(df, columns=cat_var_names, prefix_sep='=')



        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)


        Xy = get_numpy(df)

        idx = np.zeros(Xy.shape[-1]).astype(bool)
        idx[df.columns.get_loc('y')] = 1

        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)       

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        
        dtypes = df.dtypes[~idx]

        # sens idx
        sex_idx = df.columns.get_loc('age')
        race_idx = df.columns.get_loc('age')
        sens_idc = [sex_idx]


        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')
            
    elif name == 'compas':
        
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'


                # Load data
        df = pd.read_csv('/data/compas/compas-scores-two-years.csv', index_col='id', na_values=[])

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'juv_fel_count',
                  'juv_misd_count', 'juv_other_count', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score','c_charge_desc',
                 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        # ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = abs(pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count',
                'length_of_stay', 'two_year_recid','c_charge_desc']].copy()
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))

        num_train = dfcutQ.shape[0]

        num_var_names = ['two_year_recid', 'sex','race', 'score_text','priors_count', 'length_of_stay','c_charge_desc' ]
        categorical_features = ['age_cat','c_charge_degree']

        dfcutQ = pd.get_dummies(dfcutQ, columns=categorical_features, prefix_sep='=')

        for col in dfcutQ:
            if dfcutQ[col].dtype == 'object':
                dfcutQ[col] = dfcutQ[col].astype('category')
            else:
                dfcutQ[col] = dfcutQ[col].astype(float)


        pos_class_label = 1
        neg_class_label = 0

        idx = np.zeros(dfcutQ.shape[1]).astype(bool)
        y_idx = dfcutQ.columns.get_loc('two_year_recid')
        idx[y_idx] = True

        Xy = get_numpy(dfcutQ)

        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)

        #remove bad quality sample
        idx = X[:, 5] == -1

        X = X[~idx, :]
        y = y[~idx]

        dfcutQ = dfcutQ.drop(['two_year_recid'], axis = 1)
        
        dtypes = dfcutQ.dtypes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

        if verbose:
            print('Dataset split complete.')
            
        race_idx = dfcutQ.columns.get_loc('race')
        sex_idx = dfcutQ.columns.get_loc('sex')
        
        sens_idc = [race_idx, sex_idx]
        
        race_cats = dfcutQ.iloc[:,race_idx].cat.categories
        sex_cats = dfcutQ.iloc[:,sex_idx].cat.categories
        
        if verbose:
            print(race_cats, sex_cats)

        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')
    
    elif name == 'meps':
        
        df = pd.read_csv('/data/meps/h181.csv', sep=',', na_values=[])
        
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'
        
        def sex(row):
            if row['SEX'] == 1:
                return 'female'
            return 'male'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df['SEX'] = df.apply(lambda row: sex(row), axis=1)
        
        df = df.rename(columns = {'RACEV2X' : 'RACE'})

        df = df[df['PANEL'] == 19]

        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1
                
        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        
        features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
                                 'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION']
        
        categorical_features=['REGION','SEX', 'MARRY',
             'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
             'PHQ242','EMPST','POVCAT','INSCOV']
        
        df = df[features_to_keep]
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')
        
        num_train = df.shape[0]

        pos_class_label = 1
        neg_class_label = 0
        y = np.zeros(num_train)
        
        verbose = True
        
         # sens idx
        race_idx = df.columns.get_loc('RACE')
        sex_idx = df.columns.get_loc('RACE')
        
        sens_idc = [race_idx]
         
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)
                
        
        
        idx = np.zeros(df.shape[1]).astype(bool)
        y_idx = df.columns.get_loc('UTILIZATION')
        idx[y_idx] = True
        
#         min_max_scaler = MaxAbsScaler()
        Xy = get_numpy(df)
        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)
        
        df = df.drop(['UTILIZATION'], axis=1)
        dtypes = df.dtypes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # Remove sensitive information from data
        
        race_cats = df['RACE'].cat.categories
#         sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats)
            
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]


        if verbose:
            print('Senstive attribute removal complete.')

    
    elif name == 'german':
        # Download data if needed
        _german_loan_attribute_map = dict(
            A11='< 0 DM',
            A12='0-200 DM',
            A13='>= 200 DM',
            A14='no checking',
            A30='no credits',
            A31='all credits paid back',
            A32='existing credits paid back',
            A33='delayed past payments',
            A34='critical account',
            A40='car (new)',
            A41='car (used)',
            A42='furniture/equipment',
            A43='radio/television',
            A44='domestic appliances',
            A45='repairs',
            A46='education',
            A47='(vacation?)',
            A48='retraining',
            A49='business',
            A410='others',
            A61='< 100 DM',
            A62='100-500 DM',
            A63='500-1000 DM',
            A64='>= 1000 DM',
            A65='unknown/no sav acct',
            A71='unemployed',
            A72='< 1 year',
            A73='1-4 years',
            A74='4-7 years',
            A75='>= 7 years',
            #A91='male & divorced',
            #A92='female & divorced/married',
            #A93='male & single',
            #A94='male & married',
            #A95='female & single',
            A91='male',
            A92='female',
            A93='male',
            A94='male',
            A95='female',
            A101='none',
            A102='co-applicant',
            A103='guarantor',
            A121='real estate',
            A122='life insurance',
            A123='car or other',
            A124='unknown/no property',
            A141='bank',
            A142='stores',
            A143='none',
            A151='rent',
            A152='own',
            A153='for free',
            A171='unskilled & non-resident',
            A172='unskilled & resident',
            A173='skilled employee',
            A174='management/self-employed',
            A191='no telephone',
            A192='has telephone',
            A201='foreigner',
            A202='non-foreigner',
        )

        filename = '/data/german/german.data'
        if not os.path.isfile(filename):
            print('Downloading data to %s' % os.path.abspath(filename))
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                                       filename)

        # Load data and setup dtypes
        col_names = [
            'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
            'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
            'other_debtors', 'residing_since', 'property', 'age',
            'inst_plans', 'housing', 'num_credits',
            'job', 'dependents', 'telephone', 'foreign_worker', 'status']
        
#         AIF360
        column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']
        
        df = pd.read_csv(filename, delimiter=' ', header=None, names=column_names)
        
        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'
        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        def group_purpose(x):
            if x in ['A40', 'A41', 'A42', 'A43', 'A47', 'A410']:
                return 'non-essential'
            elif x in ['A44', 'A45', 'A46', 'A48', 'A49']:
                return 'essential'

        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                      'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)

        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    #     df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['purpose'] = df['purpose'].apply(lambda x:group_purpose(x))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        
        cat_features = ['credit_history', 'savings', 'employment',  'purpose', 'other_debtors', 'property', 'housing', 'skill_level', \
                'investment_as_income_percentage', 'status', 'installment_plans', 'foreign_worker']
        
        df = pd.get_dummies(df, columns=cat_features, prefix_sep='=')
        df = df.drop(['telephone', 'personal_status',], axis = 1)
    
            
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)
                
        df['age'] = df['age'].apply(lambda x: x >= 25).astype('category')

        def get_numpy(df):
            new_df = df.copy()
            cat_columns = new_df.select_dtypes(['category']).columns
            new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
            return new_df.values
        
        y_idx = df.columns.get_loc('credit')
        idx = np.zeros(df.shape[1]).astype(bool)
        idx[y_idx] = True
        
        Xy = get_numpy(df)
        X = Xy[:,~idx]
        y = Xy[:,idx].reshape(-1)
        
        # Make 1 (good customer) and 0 (bad customer)
        # (Originally 2 is bad customer and 1 is good customer)
        sel_bad = y == 2
        y[sel_bad] = 0
        y[~sel_bad] = 1
        feature_labels = df.columns.values[:-1]  # Last is prediction
        dtypes = df.dtypes[~idx]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

        # Senstivie attribute
#         foreign = 19
        age_idx = df.columns.get_loc('age')
        sex_idx = df.columns.get_loc('sex')
        sens_idc = [sex_idx, age_idx]
        
        
        age_cats = df.iloc[:, age_idx].cat.categories
        sex_cats = df.iloc[:, sex_idx].cat.categories
        print([age_cats, sex_cats])

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        race_idx = age_idx

    else:
        raise ValueError('Data name invalid.')

    return X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx

def get_csv_eqodd(fm, data_name='adult'):
    assert(fm.model is not None)
    # get csv files required for the eq_odds code to run
    label = fm.y_test
    group = fm.X_test[:, fm.sens_idx]
    prediction = fm.model.predict_proba(fm.X_test_removed)[:,1] # positive label prediction
    # make csv file
    f = open('%s_predictions.csv'%data_name, 'w')
    f.write(',label,group,prediction\n')
    for i, e in enumerate(zip(label, group, prediction)):
        line = '%d,%0.2f,%0.2f,%f\n'%(i, e[0],  e[1], e[2])
        f.write(line)
    f.close()


    
from torch.utils.data import Dataset

class tabular_dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X_raw = X
        self.Y = Y
        self.A = A
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X_raw[idx], self.Y[idx], self.A[idx]
    
    
class CelebALoader_JTT(data.Dataset):
    def __init__(self, args, annotation_dir, error_lst, split = 'train', lamda = 20, transform = None):
        print("CelebA dataloader")

        self.split = split
        self.annotation_dir = annotation_dir
        self.transform = transform

        print("loading %s annotations.........." % self.split)
        self.df = pd.read_csv(open(os.path.join(annotation_dir, split+".csv"), 'rb'))
        
        with open(error_lst) as f:
            self.error_lst = json.load(f)
            
        new_lst = []
        for err in self.error_lst:
            sample = self.df['/data/celebA/CelebA/Img/img_align_celeba/' +  err == self.df['Path']]
            for i in range(lamda):
                new_lst.append(sample)

        error_df = pd.concat(new_lst, ignore_index = True)
        self.df = pd.concat([self.df, error_df], ignore_index=True)
            
        self.labels = np.eye(2)[self.df['Attractive'].values]
        self.gender = np.eye(2)[self.df['Male'].values]
        
        print("dataset size: %d" % len(self.labels))

        self.image_dir = self.df['Path'].values
        self.image_ids = [i.split('/')[-1] for i in self.df['Path'].values]

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender[:, 1])[0]), len(np.nonzero(self.gender[:, 0])[0])))

    def __getitem__(self, index):
        image_dir = self.image_dir[index]
        
        image_ids = self.image_ids[index]
        
        img_ = Image.open(image_dir).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.labels[index]), \
                torch.LongTensor(self.gender[index]), [self.image_ids[index]]
    def __len__(self):
        return len(self.gender)
    
    
class ImSituVerbGender_JTT(data.Dataset):
    def __init__(self, args, annotation_dir, error_lst, image_dir, split = 'train', lamda = 20,  transform = None, \
        balanced_val=False, balanced_test=False):
        print("ImSituVerbGender dataloader")

        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.args = args

        verb_id_map = pickle.load(open('./verb_classification/data/verb_id.map', 'rb'))
        self.verb2id = verb_id_map['verb2id']
        self.id2verb = verb_id_map['id2verb']

        print("loading %s annotations.........." % self.split)
        self.ann_data = pickle.load(open(os.path.join(annotation_dir, split+".data"), 'rb'))

        
        print("dataset size: %d" % len(self.ann_data))
        self.verb_ann = np.zeros((len(self.ann_data), len(self.verb2id)))
        self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)
        
        args.num_verb = len(self.verb2id)

        for index, ann in enumerate(self.ann_data):
            self.verb_ann[index][ann['verb']] = 1
            self.gender_ann[index][ann['gender']] = 1

        self.image_ids = list(range(len(self.ann_data)))
        
        with open(error_lst) as f:
            self.error_lst = json.load(f)
        
        for err in self.error_lst:
            self.verb_ann = np.concatenate([self.verb_ann, self.verb_ann[err].reshape(1,-1).repeat(lamda, axis = 0)])
            self.gender_ann = np.concatenate([self.gender_ann, self.gender_ann[err].reshape(1,-1).repeat(lamda, axis = 0)])
            self.ann_data.append(self.ann_data[err])
            self.image_ids.append(err)

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender_ann[:, 0])[0]), len(np.nonzero(self.gender_ann[:, 1])[0])))
        
    def __getitem__(self, index):


        img = self.ann_data[index]
        image_name = img['image_name']
        image_path_ = os.path.join(self.image_dir, image_name)

        img_ = Image.open(image_path_).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.verb_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.LongTensor([self.image_ids[index]])

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis = 0) / (1e-15 + \
                (self.gender_ann.sum(axis = 0) + (self.gender_ann == 0).sum(axis = 0) ))

    def getVerbWeights(self):
        return (self.verb_ann == 0).sum(axis = 0) / (1e-15 + self.verb_ann.sum(axis = 0))


    def __len__(self):
        return len(self.ann_data)
