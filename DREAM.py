from lightGRN import *
import matplotlib.pyplot as plt

def Par(path,path1,path2,gold,n_estimators=0, max_depth=0,index=5,alpha=0.0114,time=50):
    TS_data = pd.read_csv(path,sep='\t').values
    SS_data_1 = pd.read_csv(path1,sep='\t').values
    SS_data_2 = pd.read_csv(path2,sep='\t').values

    # get the steady-state data
    SS_data = np.vstack([SS_data_1, SS_data_2])

    i = np.arange(0, 190, 21) #  10->85  100->190
    j = np.arange(21, 211, 21) # 10->106 100->211

    # get the time-series data
    TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]

    samples = 10
    time_points = [np.arange(0, time*20+1, time)] * samples
    ngenes = TS_data[0].shape[1]
    gene_names = ['G'+str(i+1) for i in range(ngenes)]
    regulators = gene_names.copy()

    gold_edges = pd.read_csv(gold,'\t',header=None)
    xgb_kwargs = dict(boosting_type='dart',  # goss
                      objective='regression',
                      max_depth=10,  # 7
                      num_leaves=9,  # 30
                      min_child_samples=1,  # 1
                      min_child_weight=0.001,  # 0.001
                      learning_rate=0.01,  # 0.01
                      n_estimators=398,  # 500
                      colsample_bytree=0.4,  # 0.6
                      importance_type='split',  # gain
                      #num_threads = 6
                      )
    VIM = get_importances(TS_data, time_points, alpha=0.0214, SS_data=SS_data, gene_names=gene_names, # alpha = 0.0114
                          regulators=regulators, param=xgb_kwargs)
    auroc, aupr ,precision,recall= get_scores(VIM, gold_edges, gene_names, regulators,index)
    print("AUROC:", auroc, "AUPR:", aupr)
    return auroc,aupr,precision,recall
for index in range(1,6):
    TS_data = "DREAM4/insilico_size100/insilico_size100_" + str(index) + "_timeseries.tsv"
    SS_data_1 = "DREAM4/insilico_size100/insilico_size100_" + str(index) + "_knockouts.tsv"
    SS_data_2 = "DREAM4/insilico_size100/insilico_size100_" + str(index) + "_knockdowns.tsv"
    gold = "DREAM4/insilico_size100/insilico_size100_" + str(index) + "_goldstandard.tsv"

    auroc,aupr,precision,recall = Par(TS_data, SS_data_1, SS_data_2, gold, 650, 15,index)