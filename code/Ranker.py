import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from load_data import *

class RankMSE():
    def __init__(self, id='RankMSE',gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

    def init(self):
        self.model = nn.Sequential(nn.Linear(46, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
        self.sigma = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)

    def forward(self, batch_q_doc_vectors):
        #这里的batch_size可以理解为qid的数量，num_docs则是这些qid所共有的相关文档数量(有着相同相关文档数量的qid会被划为同一个batch)，而num_features就是46
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()

        #这里向前传播完以后的结果为_batch_preds，其size是[batch_size,num_docs,1]
        _batch_preds = self.model(batch_q_doc_vectors)
        # pytorch中view函数的作用为重构张量的维度,相当于numpy中resize（）的功能
        # 这里把3维的张量转为了2维的张量，其大小为[batch_size, num_docs]
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        # print('batch_preds', batch_preds.size())
        # print(batch_preds)
        _batch_loss = F.mse_loss(batch_preds, batch_std_labels, reduction='none')
        batch_loss = torch.mean(torch.sum(_batch_loss, dim=1))

        self.optimizer.zero_grad()
        # 优化器，打分函数初始化为0
        batch_loss.backward()
        # 求打分函数参数的梯度
        self.optimizer.step()
        # 优化器内置的step、往前优化一步、省略手动的
        return batch_loss

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        batch_preds = self.forward(batch_q_doc_vectors)
        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs)

    def train(self, train_data, epoch_k=None):
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k)

            epoch_loss += batch_loss.item()

        epoch_loss = epoch_loss / num_queries
        return epoch_loss

    def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], device='cpu'):
        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.forward(batch_q_doc_vectors)
            # 将预测值转为了cpu tensor
            if self.gpu: batch_preds = batch_preds.cpu()
            # torch.sort是给tensor排序的方法
            #         # 对于二维数据：dim=0 按列排序，dim=1 按行排序，默认 dim=1
            #         # torch.sort返回两个结果，第一个是排序好后的张量，第二个是排序好的张量在原张量的对应索引
            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            # torch.gather 从原tensor中获取指定dim和指定index的数据
            #详解:https://zhuanlan.zhihu.com/p/352877584
            #这里的batch_pred_desc_inds，为预测值的从大到小索引，batch_std_labels的大小也为[batch_size, num_docs]
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

            batch_ideal_rankings = batch_std_labels

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks

class RankNet():
    def __init__(self, id='RankNet',gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

    def init(self):
        self.model = nn.Sequential(nn.Linear(46, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
        self.sigma = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)

    def forward(self, batch_q_doc_vectors):
        #这里的batch_size可以理解为qid的数量，num_docs则是这些qid所共有的相关文档数量(有着相同相关文档数量的qid会被划为同一个batch)，而num_features就是46
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()

        #这里向前传播完以后的结果为_batch_preds，其size是[batch_size,num_docs,1]
        _batch_preds = self.model(batch_q_doc_vectors)
        # pytorch中view函数的作用为重构张量的维度,相当于numpy中resize（）的功能
        # 这里把3维的张量转为了2维的张量，其大小为[batch_size, num_docs]
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        #计算某个文档排在某个文档前的概率值
        batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        #batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        #得到标准的概率值
        batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        #损失函数
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        batch_preds = self.forward(batch_q_doc_vectors)
        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs)

    def train(self, train_data, epoch_k=None):
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k)

            epoch_loss += batch_loss.item()

        epoch_loss = epoch_loss / num_queries
        return epoch_loss

    def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], device='cpu'):
        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.forward(batch_q_doc_vectors)
            # 将预测值转为了cpu tensor
            if self.gpu: batch_preds = batch_preds.cpu()
            # torch.sort是给tensor排序的方法
            #         # 对于二维数据：dim=0 按列排序，dim=1 按行排序，默认 dim=1
            #         # torch.sort返回两个结果，第一个是排序好后的张量，第二个是排序好的张量在原张量的对应索引
            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            # torch.gather 从原tensor中获取指定dim和指定index的数据
            #详解:https://zhuanlan.zhihu.com/p/352877584
            #这里的batch_pred_desc_inds，为预测值的从大到小索引，batch_std_labels的大小也为[batch_size, num_docs]
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

            batch_ideal_rankings = batch_std_labels

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks


def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings, device='cpu'):
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, device=device)

    batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0

    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_predict_rankings.size(1), dtype=torch.float, device=device)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg

class LambdaRank():
    def __init__(self, id='LambdaRank', gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device
        self.sigma = 1.0
    def init(self):
        self.model = nn.Sequential(nn.Linear(46, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
        self.sigma = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)

    def forward(self, batch_q_doc_vectors):
        #这里的batch_size可以理解为qid的数量，num_docs则是这些qid所共有的相关文档数量(有着相同相关文档数量的qid会被划为同一个batch)，而num_features就是46
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()

        #这里向前传播完以后的结果为_batch_preds，其size是[batch_size,num_docs,1]
        _batch_preds = self.model(batch_q_doc_vectors)
        # pytorch中view函数的作用为重构张量的维度,相当于numpy中resize（）的功能
        # 这里把3维的张量转为了2维的张量，其大小为[batch_size, num_docs]
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        # 根据模型的预测值对文档进行从大到小的排序(按行排序)
        batch_descending_preds, batch_pred_desc_inds = torch.sort(batch_preds, dim=1,
                                                                  descending=True)  # sort documents according to the predicted relevance
        # 对标签也相应的排序，进行对应
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1,
                                              index=batch_pred_desc_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        # torch.unsqueeze
        # 作用：扩展维度
        # 返回一个新的张量，对输入的既定位置插入维度 1
        # batch_predict_rankings.size()=[batch, ranking_size]
        # torch.unsqueeze(batch_predict_rankings, dim=2)→[batch, ranking_size, 1] torch.unsqueeze(batch_predict_rankings, dim=1)→[batch, 1, ranking_size]
        # batch_std_diffs.size()=[batch, ranking_size, ranking_size]
        batch_std_diffs = torch.unsqueeze(batch_predict_rankings, dim=2) - torch.unsqueeze(batch_predict_rankings,
                                                                                           dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_std_Sij)

        batch_s_ij = torch.unsqueeze(batch_descending_preds, dim=2) - torch.unsqueeze(batch_descending_preds,
                                                                                      dim=1)  # computing pairwise differences, i.e., s_i - s_j
        # batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                          batch_predict_rankings=batch_predict_rankings, device=self.device)

        # a direct setting of reduction='mean' is meaningless due to breaking the query unit, which also leads to poor performance
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1),
                                             weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')
        # 把张量转换成标量，保证batch_loss是一个标量（没有方向的数值）
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()

        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        batch_preds = self.forward(batch_q_doc_vectors)
        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs)

    def train(self, train_data, epoch_k=None):
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k)

            epoch_loss += batch_loss.item()

        epoch_loss = epoch_loss / num_queries
        return epoch_loss

    def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], device='cpu'):
        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.forward(batch_q_doc_vectors)
            # 将预测值转为了cpu tensor
            if self.gpu: batch_preds = batch_preds.cpu()
            # torch.sort是给tensor排序的方法
            #         # 对于二维数据：dim=0 按列排序，dim=1 按行排序，默认 dim=1
            #         # torch.sort返回两个结果，第一个是排序好后的张量，第二个是排序好的张量在原张量的对应索引
            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            # torch.gather 从原tensor中获取指定dim和指定index的数据
            #详解:https://zhuanlan.zhihu.com/p/352877584
            #这里的batch_pred_desc_inds，为预测值的从大到小索引，batch_std_labels的大小也为[batch_size, num_docs]
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

            batch_ideal_rankings = batch_std_labels

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks