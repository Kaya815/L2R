import datetime
from Ranker import *

def metric_results_to_string(list_scores=None, list_cutoffs=None, split_str=', ', metric='nDCG'):
    """
    Convert metric results to a string representation
    :param list_scores:
    :param list_cutoffs:
    :param split_str:
    :return:
    """
    list_str = []
    for i in range(len(list_scores)):
        list_str.append(metric + '@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)

def evaluation(data_id=None, dir_data=None, model_id=None, batch_size=100):
    """
    如果有k个fold，通过交叉验证来评估学习排名方法，否则仅用一个fold。
    Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
    :param data_dict:       settings w.r.t. data
    :param eval_dict:       settings w.r.t. evaluation
    :param sf_para_dict:    settings w.r.t. scoring function
    :param model_para_dict: settings w.r.t. the ltr_adhoc model
    :return:
    """
    fold_num = 5
    cutoffs = [1, 3, 5, 10]
    epochs = 10 #训练10轮
    ranker = globals()[model_id]()

    time_begin = datetime.datetime.now()       # timing
    l2r_cv_avg_scores = np.zeros(len(cutoffs)) # fold average

    #1-5
    for fold_k in range(1, fold_num + 1):   # evaluation over k-fold data
        ranker.init()           # initialize or reset with the same random initialization

        train_data, test_data = load_multiple_data(data_id=data_id, dir_data=dir_data, fold_k=fold_k)
        #test_data = None

        for epoch_k in range(1, epochs + 1):
            torch_fold_k_epoch_k_loss = ranker.train(train_data=train_data, epoch_k=epoch_k)

        torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, device='cpu')
        fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()

        performance_list = [model_id + ' Fold-' + str(fold_k)]      # fold-wise performance
        for i, co in enumerate(cutoffs):
            performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_ndcg_ks[i]))
        performance_str = '\t'.join(performance_list)
        print('\t', performance_str)

        l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks) # sum for later cv-performance

    time_end = datetime.datetime.now()  # overall timing
    elapsed_time_str = str(time_end - time_begin)
    print('Elapsed time:\t', elapsed_time_str + "\n\n")

    l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
    eval_prefix = str(fold_num) + '-fold average scores:'
    print(model_id, eval_prefix, metric_results_to_string(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))  # print either cv or average performance

    return l2r_cv_avg_scores


data_id  = 'MQ2008_Super'
dir_data = 'D:/Data/MQ2008/'

model_id = 'RankMSE' # RankMSE, RankNet, LambdaRank
print(model_id)
evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)

model_id = 'LambdaRank' # RankMSE, RankNet, LambdaRank
print(model_id)
evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)

model_id = 'RankNet' # RankMSE, RankNet, LambdaRank
print(model_id)
evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)
