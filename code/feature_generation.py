import pandas as pd
import numpy as np
import data_preprocess


def feature_agg_window(df, DeltaT=[-30,-7,-3,-1,1,3,7,30], prefix=''):
    '''
    aggregate features for each event 
        A: count the events happended between T and T+DeltaT
        B: sum the fatalities happended between T and T+DeltaT
    '''
    # initialize continuous day
    day_range = pd.date_range('1997-01-01', '2023-02-10')
    # initialize with 0 for day count and fatality sum
    day_ctr = pd.Series([0]*len(day_range),index = day_range)
    fatalities_sum = pd.Series([0]*len(day_range),index = day_range)
    day_ctr.index.name, fatalities_sum.index.name = 'event_date', 'event_date'
    # fill event counts from data
    event_count = df.groupby('event_date')['data_id'].agg('count')
    day_ctr[event_count.index] = event_count
    # fill fatalities sum from data
    fatal_sum   = df.groupby('event_date')['fatalities'].agg('sum')
    fatalities_sum[fatal_sum.index] = fatal_sum
    out = []
    # aggregate for event count
    for delta in DeltaT:
        if delta<0:
            event_count_agg = day_ctr.rolling(-delta, center=False, closed='left').sum()
        else:
            event_count_agg = day_ctr.shift(-1).rolling(delta, center=False, closed='left').sum().shift(-delta)
        out.append(event_count_agg.copy())
    # aggregate for fatality sum
    for delta in DeltaT:
        if delta<0:
            fatalities_sum_agg = fatalities_sum.rolling(-delta, center=False, closed='left').sum()
        else:
            fatalities_sum_agg = fatalities_sum.shift(-1).rolling(delta, center=False, closed='left').sum().shift(-delta)
        out.append(fatalities_sum_agg.copy())
    
    out_frame = pd.concat(out, axis=1).reset_index()
    out_frame.columns = ['event_date'] + \
                        ['event_ctr_' + prefix + str(delta) for delta in DeltaT] + \
                        ['fatality_sum_' + prefix + str(delta) for delta in DeltaT]

    df = pd.merge(df, out_frame, how='left', on='event_date')

    return df


def generate_feature(data):
    '''
    This function generate economic interesting features 
    DeltaT is chosen among [-30,-7,-3,-1,1,3,7,30]
    For each event happened at time T, space S, involved actor1 A1 and actor2 A2, interaction code I we generate the following features 
        1. #events / sum(fatalities) happended between T and T+DeltaT  
        2. #events / sum(fatalities) happended between T and T+DeltaT,  at the same admin1 level 
        3. #events / sum(fatalities) happended between T and T+DeltaT,  at the same admin2 level 
        4. #events / sum(fatalities) happended between T and T+DeltaT,  for the actor1 A1
        5. #events / sum(fatalities) happended between T and T+DeltaT,  for the actor1 A1 and under same interaction level I, 

    Note: if DeltaT is negative, we look into a historical window [T-DeltaT, T]; if DeltaT is positive, we look into a future window [T, T+DeltaT]
    '''
    key_feat = ['data_id', 'event_date', 'fatalities']
    prefix_list = ['', 'admin1_', 'admin2_', 'actor1_', 'actor1_inter_']
    # 1 
    event_ctr = feature_agg_window(data[key_feat], prefix=prefix_list[0])

    # 2
    admin1_event_ctr = data.groupby('admin1')[key_feat].apply(lambda x: feature_agg_window(x, prefix=prefix_list[1])).droplevel(0)

    # 3
    admin2_event_ctr = data.groupby(['admin1', 'admin2'])[key_feat].apply(lambda x: feature_agg_window(x, prefix=prefix_list[2])).droplevel(0)

    # 4
    actor1_event_ctr = data.groupby('actor1')[key_feat].apply(lambda x: feature_agg_window(x, prefix=prefix_list[3])).droplevel(0)

    # 5 
    actor1_inter_event_ctr = data.groupby(['actor1', 'interaction'])[key_feat].apply(lambda x: feature_agg_window(x, prefix=prefix_list[4])).droplevel(0)

    # merge everybody
    data = data.merge(event_ctr, on = key_feat, how='left')
    data = data.merge(admin1_event_ctr, on=key_feat, how='left')
    data = data.merge(admin2_event_ctr, on=key_feat, how='left')
    data = data.merge(actor1_event_ctr, on=key_feat, how='left')
    data = data.merge(actor1_inter_event_ctr, on=key_feat, how='left')

    return data


if __name__ == '__main__':
    data = pd.read_csv('../data/data.csv')
    data = data_preprocess.pre_process_interp(data)
    data_feat = generate_feature(data)
    data_feat.to_csv('../data/data_feat.csv', index=False)