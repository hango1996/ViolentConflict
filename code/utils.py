import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ClusterDisplay(col, data_cluster):
    _, ax=plt.subplots(1, 1, figsize=[30,10])
    label_order = data_cluster[col].groupby('cluster_label').mean().sort_values().index
    subdata = data_cluster[col].reset_index()
    sns.boxplot(data=subdata, x='cluster_label', y=col, order=label_order, ax=ax)
    plt.show()


def ClusterPctChange(data_cluster):
    data_cluster_mean = data_cluster.groupby('cluster_label').mean().stack().reset_index()
    data_cluster_mean.columns = ['cluster_label', 'feature_window', 'average']
    data_cluster_mean['feature'] = data_cluster_mean['feature_window'].map(lambda x: x[:-3] if x[-3]=='_' else x[:-4])
    data_cluster_mean['window']  = data_cluster_mean['feature_window'].map(lambda x: x[-2:] if x[-3]=='_' else x[-3:])
    data_cluster_mean.drop('feature_window', axis=1, inplace=True)
    data_cluster_mean = data_cluster_mean.set_index(['cluster_label', 'feature', 'window'])
    data_cluster_mean = data_cluster_mean.unstack(level=2)
    data_cluster_mean.dropna(axis=0, inplace=True)
    data_cluster_mean.columns = ['-30','30']
    data_cluster_mean['pct_change'] = data_cluster_mean['30'] / data_cluster_mean['-30'] - 1
    data_cluster_mean = data_cluster_mean.sort_values(['feature', 'pct_change'], ascending=[True, False]).reset_index(level=0)

    return data_cluster_mean.groupby('feature').apply(lambda x: pd.DataFrame(x.iloc[[0,-1]].values, 
                                                                            index=['max', 'min'], 
                                                                            columns=['cluster_label', '-30', '30', 'pct_change']))


def ConstructCluster(data_feat, data_cluster):
    feat_col = ['data_id'] + [data_feat.columns[i] for i in range(31, 54)]

    data_cluster = data_cluster.merge(data_feat[feat_col], on='data_id', how='left')
    data_cluster['cluster_label'] = data_cluster['cluster_label'].astype('object')

    data_cluster = data_cluster.set_index(['data_id', 'cluster_label'])
    data_cluster.reset_index(level=0, drop=True, inplace=True)    
    return data_cluster