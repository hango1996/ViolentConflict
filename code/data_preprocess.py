import pandas as pd
import numpy as np

def pre_process_model(data0):
    data = data0[['data_id', 'event_date', 'event_type', 'interaction', 'region', 'country', 'latitude', 'longitude', 'fatalities']].copy()
    data['event_date'] = pd.to_datetime(data.event_date)
    data['month-year'] = data['event_date'].dt.strftime('%m-%y')
    data['event_month_ctr'] = data.groupby('month-year')['data_id'].transform('count')
    data['transformed_fatalities'] = np.log(data['fatalities']+1)
    data['days_from_19970101'] = (data['event_date'] - pd.to_datetime('1997-01-01')).dt.days
    data.drop(['data_id', 'event_date', 'fatalities', 'month-year'], inplace=True, axis=1)
    data['interaction'] = data['interaction'].astype('str')

    return data

def pre_process_interp(data):
    data['event_date'] = pd.to_datetime(data.event_date)
    return data


if __name__ == '__main__':
    data = pd.read_csv('../data/data.csv')
    pre_process_model(data)
