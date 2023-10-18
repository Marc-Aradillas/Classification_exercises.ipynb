import pandas as pd
import acquire

from sklearn.model_selection import train_test_split

#function for droppping columns

def drop_cols(df):
    
    return df.drop(columns = ['pclass', 'passenger_id', 'embarked', 'deck'])


def prep_iris():
    
    iris_df = acquire.get_iris_data()
    
    i_df = iris_df.drop(columns=['species_id', 'species_id.1', 'measurement_id'])

    i_df = i_df.rename(columns={'species_name' : 'species'})
    
    # Create dummy variables for the 'species' column
    species_dummies = pd.get_dummies(i_df['species'], prefix='').astype(int)
    
    # i_df = i_df.drop(columns = 'species')
    
    # Concatenate the dummy variables onto the original dataframe
    i_df = pd.concat([i_df, species_dummies], axis=1)
    
    return i_df



def prep_titanic():

    titanic_data = acquire.get_titanic_data()
    
    t_data = titanic_data.drop(columns= ['passenger_id', 'embarked', 'deck'])

    return t_data


def prep_telco():

    telco_data = acquire.get_telco_data()

    te_data = telco_data.drop(columns=['customer_id', 'payment_type_id', 'contract_type_id', 'internet_service_type_id'])

    return te_data    
    

def train_val_test(df, strat, seed = 42):

    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed,
                                       stratify = df[strat])

    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed,
                                 stratify = val_test[strat])

    return train, val, test


# ------------------------------------------------------------------------------------------


#defined function to impute vals
def impute_vals(train, val, test):
    
    town_mode = train.embark_town.mode()
    
    train.embark_town = train.embark_town.fillna(town_mode)
    val.embark_town = val.embark_town.fillna(town_mode)
    test.embark_town = test.embark_town.fillna(town_mode)
    
    med_age = train.age.median()
    
    train.age = train.age.fillna(med_age)
    val.age = val.age.fillna(med_age)
    test.age = test.age.fillna(med_age)
    
    return train, val, test



def dummies(df):

    df = pd.get_dummies(df, columns = ['sex'], drop_first = True)
        
    return df
    


def iris_pipeline():
    
    df = prep_iris()
    
    train, val, test = train_val_test(df, 'species')
    
    return train, val, test


def titanic_pipeline():
    
    df = prep_titanic()
    
    train, val, test = train_val_test(df, 'survived')

    train, val, test = impute_vals(train, val, test)

    train = dummies(train)
    
    val = dummies(val)
    
    test = dummies(test)
    
    return train, val, test


#defined function for telco_ ataset
def telco_pipeline():
    
    df = prep_telco()
    
    train, val, test = train_val_test(df, 'churn')
    
    return train, val, test



def eval_p(p, a = 0.05):

    if p < a:

        print(f'The result is significant, we reject the null hypothesis with a p-value of {round(p, 2)}.')

    else:

        print(f'We failed to reject the null hypothesis with a p-value of {round(p, 2)}.')
