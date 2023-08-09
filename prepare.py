import pandas as pd
import acquire

from prepare import train_val_test




def prep_iris():

    iris_df = acquire.get_iris_data()

    i_df = iris_df.drop(columns=['species_id', 'species_id.1', 'measurement_id'])

    i_df = i_df.rename(columns={'species_name' : 'species'})

    return i_df




def prep_titanic():

    titanic_data = acquire.get_titanic_data()
    
    t_data = titanic_data.drop(columns= 'embarked')

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



def eval_p(p, a = 0.05):

    if p < a:

        print(f'The result is significant, we reject the null hypothesis with a p-value of {round(p, 2)}.')

    else:

        print(f'We failed to reject the null hypothesis with a p-value of {round(p, 2)}.')