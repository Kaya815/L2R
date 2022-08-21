import torch.utils.data as data
from metric_ndcg import *

def np_arg_shuffle_ties(vec):

    #如果传入的不是向量，这个函数便无法运作
    if len(vec.shape) > 1:
        raise NotImplementedError
    else:
        #length获得了传入向量的大小，在这里实际就是同一个qid之下的文档标准值个数
        length = vec.shape[0]
        #np.random.permutation是随机排列一个数组的方法
        #它可以接受一个整数或者数组。若参数为整数，则先由该整生数成一个np.arange(x)然后再随机排序
        #且当输入是一个多维数组时，仅会根据第一个维度进行随机排序
        #所以这里得到的是一个大小为同一个qid之下的文档标准值个数的随机排序数组
        perm = np.random.permutation(length)
        #对MQ2008_super而言，descending为True
        #这里的-vec[perm]是先根据刚刚随机得到的顺序数组perm，将标签值vec重新排序。接着在将所有的标签值变为相反数
        #np.argsort是返回一个数组从小到大排序后各元素原始下标(这里的原始下标就是-vec[perm]中的下标)的函数
        sorted_shuffled_vec_inds = np.argsort(-vec[perm])

        #这个shuffle_ties_inds是将sorted_shuffled_vec_inds作为索引，输出了perm所对应的元素值。
        shuffle_ties_inds = perm[sorted_shuffled_vec_inds]
        return shuffle_ties_inds

def _parse_docid(comment):
    #以'MQ2008_Super'为例这里输入的comment应该是这种形式的docid = GX004-93-7097963 inc = 0.0428115405134536 prob = 0.860366
    #得到的parts=['docid', '=', 'GX004-93-7097963', 'inc', '=', '0.0428115405134536', 'prob', '=', '0.860366']
    parts = comment.strip().split()
    #因此parts[2]就是GX004-93-7097963
    return parts[2]

def clip_query_data(qid, list_docids=None, feature_mat=None, std_label_vec=None, min_docs=None, min_rele=1):
    # 这里传入的feature_mat是一个大小为(某一个qid下文档数量,每个文档特征量数量即46)的数组
    #因此feature_mat.shape[0]就是指某一个qid之下的所有文档数量，这个if会跳过那些文档数量少于设定值min_docs的qid
    if feature_mat.shape[0] < min_docs: # skip queries with documents that are fewer the pre-specified min_docs
        return None
        #这里传入的std_label_vec是一个大小为(某一个qid下文档数量)的向量数组,其含义为某一个qid下所有文档的标准值合集
        #std_label_vec > 0就是与query有相关度的文档，(std_label_vec > 0).sum()则是某一个qid下所有有相关度的文档的总数
        #这个if就是判断某一qid下，所有相关文档的总数是否小于设定值(min_rele，即最小相关文档数量)
    if (std_label_vec > 0).sum() < min_rele:
        # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
        return None

    #这个des_inds是根据文档相关度的降序(从大到小)得到的index
    des_inds = np_arg_shuffle_ties(std_label_vec)  # sampling by shuffling ties
    #这里最终得到的feature_mat, std_label_vec便是同一个qid下，根据文档相关度的降序预排序后的文档特征量矩阵和文档标准值向量
    feature_mat, std_label_vec = feature_mat[des_inds], std_label_vec[des_inds]

    return (qid, feature_mat, std_label_vec)

def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0):
    for line in lines:
        #partition()是根据指定字符分割字符串的方法。返回一个3元的元组，第一个为分隔符左边的子串，第二个为分隔符本身，第三个为分隔符右边的子串。
        #rstrip()是删除字符串末尾空格的方法。但是对于此处的line而言应该没有实际意义
        #最终data是包含了target值，qid值，46个特征量值的字符串。_为'#'号，而comment则是评论。
        data, _, comment = line.rstrip().partition('#')
        #split()在默认情况下根据空字符串对指定字符串进行分割。并且最终返回被分割的字符串列表。
        #所以此处的toks为[target值,qid值,1:特征量1，2:特征量2,...46:特征量46]
        toks = data.split()

        #对特征量个数进行初始化
        num_features = 0
        #np.repeat是重复某一元素指定次数的方法。np.repeat(元素，重复次数)
        #所以这里得到的是feature_vec=[0. 0. 0. 0. 0. 0. 0. 0.]
        feature_vec = np.repeat(missing, 8)

        #从toks里提取出了标准值
        std_score = float(toks[0])
        #更新toks为[qid值,1:特征量1，2:特征量2,...46:特征量46]
        toks = toks[1:]

        qid =toks[0][4:]

        #这个循环是从toks里面将特征值作为数值分离出来
        for tok in toks[1:]:
            #fid就是1到46这些数字，_是分隔符也就是':'，val是具体的特征量值
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                #这里将fid变为了从0-45的数字
                fid -= 1

            #这里的len(feature_vec)最初为8，循环到第8次，也就是fid等于8时，第一次进入这个while中。
            while len(feature_vec) <= fid:
                orig = len(feature_vec)
                #resize是Numpy里面重置数组大小的方法
                feature_vec.resize(len(feature_vec) * 2)
                feature_vec[orig:orig * 2] = missing

            #总之通过以上和以下两行代码，他把每行的46个特征值都按顺序放入到了feature_vec这个数组中
            feature_vec[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        feature_vec.resize(num_features)

        yield (feature_vec, std_score, qid, comment)

def parse_letor(source, has_targets=True, one_indexed=True, missing=0.0):

    max_width = 0
    #初始化X，y，qids
    feature_vecs, std_scores, qids, comments = [], [], [],[]

    # 得到了包含了每一行feature_vec，std_score，qid, comment的元组
    it = iter_lines(source, has_targets=has_targets, one_indexed=one_indexed, missing=missing)
    for f_vec, s, qid, comment in it:
        #因为it是元组里面的feature_vec，std_score，qid, comment不能被修改，所以这里把它们分别添加到事前准备好的列表里，方便之后的处理
        feature_vecs.append(f_vec)
        std_scores.append(s)
        qids.append(qid)
        comments.append(comment)
        #这里f_vec应该为46，所以max_width也被更新为了46
        max_width = max(max_width, len(f_vec))

    #这里用np.ndarray创建了一个数组
    #len(feature_vecs)为某一fold中训练数据的行数，而max_width为每行中特征量的个数，即46
    all_features_mat = np.ndarray((len(feature_vecs), max_width), dtype=np.float64)
    #先用0.0将刚刚创建好的特征量矩阵(数组)填满
    all_features_mat.fill(missing)
    #这里的feature_vecs应该是包含了每一行特征值的列表，其内部结构应该为[[第1行的特征量],[第2行的特征量],...,[第n行的特征量]]
    #enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    #这里的i就是下标，其实也就代表了是第几行的特征值，而x就是每行的46特征值所组成的数组
    for i, x in enumerate(feature_vecs):
        #这里应该是把每一行的特征量全部填入了刚刚预设好的矩阵中
        all_features_mat[i, :len(x)] = x

    #这里再获得了包含每一行目标值的向量(将列表转为了数组)
    all_labels_vec = np.array(std_scores)

    #docids是一个包含了每一行comment id的列表，实际为[GX004-93-7097963,GX010-40-4497720,GX016-32-1454614....]
    docids = [_parse_docid(comment) for comment in comments]
    #features, std_scores, qids, docids
    #all_features_mat是大小为[train.txt行数，46]的矩阵，all_labels_vec是大小为[train.txt行数]的向量，qids和docids则是len为train.txt行数的列表
    return all_features_mat, all_labels_vec, qids, docids

def iter_queries(in_file, presort=None, data_dict=None):

    #min_docs=1,min_rele=1
    min_docs, min_rele = data_dict['min_docs'], data_dict['min_rele']
    list_Qs = []
    print(in_file)
    #用open函数打开了'D:/Data/MQ2008/Fold1/train.txt'，也就是测试数据文件
    with open(in_file, encoding='iso-8859-1') as file_obj:
        #创建了一个空的dict_data字典
        dict_data = dict()
        # all_features_mat是大小为[train.txt行数，46]的矩阵，all_labels_vec是大小为[train.txt行数]的向量，qids和docids则是len为train.txt行数的列表
        all_features_mat, all_labels_vec, qids, docids = parse_letor(file_obj.readlines())

        #这个for循环又把每一行的特征量，标准值，qid，docid又取了出来
        for i in range(len(qids)):
            f_vec = all_features_mat[i, :]
            std_s = all_labels_vec[i]
            qid = qids[i]
            docid = docids[i]

            #这里的if函数是把qid相同的行的标准值，docid，特征值放在了字典dict_data中
            #最终得到的dict_data的形式为{qid0:[(std_s0, 'docid0', [f_vec01...f_vec046]),(std_s1, 'docid1', [f_vec11...f_vec146])],qid1:[(std_s2, 'docid3', [f_vec21...f_vec246]),(std_s3, 'docid4', [f_vec31...f_vec346])]....}
            #类似于{0: [(1, 'x', [0, 1, 1]), (0, 'y', [0, 1, 2])], 1: [(1, 's', [0, 1, 0]), (2, 'm', [0, 1, 1])]}
            if qid in dict_data:
                dict_data[qid].append((std_s, docid, f_vec))
            else:
                dict_data[qid] = [(std_s, docid, f_vec)]

        del all_features_mat
        # unique qids
        #set() 函数创建一个无序不重复元素集
        seen = set()
        #add() 方法用于给集合添加元素，如果添加的元素在集合中已存在，则不执行任何操作。
        seen_add = seen.add
        # sequential unique id
        #这一行我没太懂，应该是将qids唯一序列化？得到的qids_unique应该是包含了所有不重复的qid的列表
        qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

        for qid in qids_unique:
            #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            #在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
            #返回的对象或者列表是根据不同列表的index一一对应的到的
            #以刚刚的{0: [(1, 'x', [0, 1, 1]), (0, 'y', [0, 1, 2])], 1: [(1, 's', [0, 1, 0]), (2, 'm', [0, 1, 1])]}为例
            #tmp为[(1, 2), ('s', 'm'), ([0, 1, 0], [0, 1, 1])]
            #也就是说tmp把同一个qid之下的所有文档的标准值，commentid，特征量都分别打包，并且合并为了一个列表。
            tmp = list(zip(*dict_data[qid]))

            #这个list_labels_per_q的内容和名字一样，就是同一个qid之下的所有文档的标准值列表
            list_labels_per_q = tmp[0]

            #list_labels_per_q则是同一个qid之下的所有文档的特征值列表
            list_features_per_q = tmp[2]
            #vstack是按照行的顺序将元组，列表，或者numpy数组堆叠起来起来的函数
            #这里最终得到的feature_mat将会是一个大小为(某一个qid下文档数量,每个文档特征量数量即46)的数组
            #以刚刚的tmp为[(1, 2), ('s', 'm'), ([0, 1, 0], [0, 1, 1])]为例feature_mat为[[0 1 1][0 1 2]]，且大小为(2, 3)
            feature_mat = np.vstack(list_features_per_q)

            #由于上面的操作clip_query在这里为True
            #因为MQ2008_super需要预排序，所以这里clip_query_data最终返回的是(qid,降序后的文档特征量矩阵，降序后的文档标准值)这样一个元组
            Q = clip_query_data(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q),min_docs=min_docs, min_rele=min_rele)
            if Q is not None:
                #这里就最终得到了包含了所有预排序好的，每个qid之下的(qid,文档特征量矩阵，文档标准值)列表！
                list_Qs.append(Q)

    return list_Qs

class LTRDataset(data.Dataset):
    """
    Loading the specified dataset as torch.utils.data.Dataset.
    We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    """
    #必须的参数是split_type即数据分割方式，file即训练或者测试数据，data_id或者data_dict即数据集的index，batch_size即批处理数据数量
    def __init__(self, file, data_id=None, data_dict=None, presort=True,batch_size=1):
        """
        这里通过调用get_default_data_dict函数获得了包含了数据集所有基本参数的字典data_dict
        以MQ2008_super为例，其data_dict的具体内容如下
        data_dict = dict(data_id='MQ2008_Super', min_docs=1, min_rele=1, binary_rele=False,unknown_as_zero=False,
                          train_presort=True, validation_presort=True, test_presort=True,
                          train_batch_size=100, validation_batch_size=100, test_batch_size=100,
                          scale_data=False, scaler_id=None, scaler_level=None,
                          max_rele_level = 2, label_type = LABEL_TYPE.MultiLabel, num_features = 46, has_comment = True, fold_num = 5)
        """
        ''' split-specific settings '''
        self.presort = presort
        self.data_id = data_dict['data_id']
        self.list_torch_Qs = []

        #这里得到了包含了所有预排序好的，每个qid之下的(qid,文档特征量矩阵(数组)，文档标准值(数组))列表！
        list_Qs = iter_queries(in_file=file, presort=self.presort, data_dict=data_dict)
        #len(list_Qs)指的是不同的qid的总数
        #list_inds则是用这个qid总数生成了一个index列表
        list_inds = list(range(len(list_Qs)))
        for ind in list_inds:
            #这个循环从list_Qs中把qid，各个文档特征量doc_reprs(数组，大小为(qid下文档数量，46))，各个文档标准值(数组，大小为qid下文档数量)都提取出来了doc_labels
            qid, doc_reprs, doc_labels = list_Qs[ind]
            #将特征量数组转为了张量
            torch_q_doc_vectors = torch.from_numpy(doc_reprs).type(torch.FloatTensor)
            #torch_q_doc_vectors = torch.unsqueeze(torch_q_doc_vectors, dim=0)  # a default batch size of 1
            #将标准值向量转为了张量
            torch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)
            #torch_std_labels = torch.unsqueeze(torch_std_labels, dim=0) # a default batch size of 1

            #最终获得了张量列表，其基本结构和list_Qs保持一致。每个qid之下的(qid,文档特征量矩阵(张量)，文档标准值(张量))列表
            self.list_torch_Qs.append((qid, torch_q_doc_vectors, torch_std_labels))
        print('Num of q:', len(self.list_torch_Qs))

    def get_default_data_dict(self, data_id , batch_size=None):
        ''' a default setting for loading a dataset '''
        min_docs = 1 #一个数据集包含的最小文档数量
        min_rele = 1 #为-1的话意味着不关心没有相关文档的哑巴查询。例如，为了检查原始数据集的统计数据。

        #MQ2008_super不在MSLETOR_SEMI里面所以这里的train_presort为True
        #presort应该就是指定训练前是否预先排序
        train_presort = True

        #在没有指定batch_size时默认batch_size为10
        batch_size = 10 if batch_size is None else batch_size

        #这里用dict函数获得了包含各种数据信息的data_dict
        data_dict = dict(data_id=data_id, min_docs=min_docs, min_rele=min_rele, train_presort=train_presort, validation_presort=True, test_presort=True,
                         train_batch_size=batch_size, validation_batch_size=batch_size, test_batch_size=batch_size)

        return data_dict

    def __len__(self):
        #定义一下魔法函数，调用_len_的时候就返回list_torch_Qs的长度
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        #这个也是定义一下__getitem__的含义，给定一个index就返回对应list_torch_Qs的内容
        #getitem这个魔法函数在pre_allocate_batch函数中被调用了的
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        return qid, torch_batch_rankings, torch_batch_std_labels


def pre_allocate_batch(dict_univ_bin, num_docs_per_batch):
    list_batch_inds = []

    # 导入的dict_univ_bin的形式为:{文档数量x:[对应qid在list_num_docs中的索引],...}
    #univ为不同qid的相同文档总数，bin为不同qid在list_num_docs中的索引，bin_length则为有着相同文档数量的qid总数
    #例如dict_univ_bin={2: [0, 1], 3: [2]}，则有univ 2 bin [0, 1] bin_length 2univ 3 bin [2] bin_length 1
    for univ in dict_univ_bin:
        bin = dict_univ_bin[univ]
        bin_length = len(bin)

        #num_docs_per_batch可以理解为100
        #univ * bin_length为相同文档总数乘以qid总数，如果这个值没有超过预设的batch大小，就把他们打包为同一个batch
        if univ * bin_length < num_docs_per_batch: # merge all queries as one batch
            #这里的list_batch_inds是被打包在一起的qid索引
            #list_batch_inds [[0, 1], [2]]
            list_batch_inds.append(bin)
            #乘积超过了预设的batch大小，但对于MQ2008_super而言并没有这种情况，所以这一段又可以跳过
        else:
            #如果有着相同qid的文档总数小于batcsize
            if univ < num_docs_per_batch: # split with an approximate value
                num_inds_per_batch = num_docs_per_batch // univ
                for i in range(0, bin_length, num_inds_per_batch):
                    sub_bin = bin[i: min(i+num_inds_per_batch, bin_length)]
                    list_batch_inds.append(sub_bin)
            else: # one single query as a batch
                    for index in bin:
                        single_ind_as_batch = [index]
                        list_batch_inds.append(single_ind_as_batch)

    return list_batch_inds

class LETORSampler(data.Sampler):
    '''
    为LETOR数据集定制的采样器。基于以下观察，虽然每个查询的文件数量可能不同，但有许多查询的文件数量是相同的，特别是在大数据集上。
    Customized sampler for LETOR datasets based on the observation that:
    though the number of documents per query may differ, there are many queries that have the same number of documents, especially with a big dataset.
    '''
    #这里传入的data_source其实是一个LTRDataset对象
    def __init__(self, data_source, rough_batch_size=None):
        list_num_docs = []
        #由于LTRDataset中定义了_getitem_函数，所以LTRDataset对象可以用类似列表的方式调用
        #给定一个index就返回对应list_torch_Qs的内容
        #qid, torch_batch_rankings(也就是文档群的特征值), torch_batch_std_labels = self.list_torch_Qs[index]
        for qid, torch_batch_rankings, torch_batch_std_labels in data_source:
            #这里是在统计每个qid下面有多少个文档，并且将统计的结果添加到list_num_docs列表中
            #torch_batch_std_labels.size(0)等价于qid下的文档数
            list_num_docs.append(torch_batch_std_labels.size(0))

        dict_univ_bin = {}
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        #这里的ind就是list_num_docs中元素的索引，univ则是list_num_docs中的元素(也就是每个qid下的文档数量)
        #这个循环的目的是将有相同文档数量的qid都组合在了一起并且添加到了字典dict_univ_bin中去
        #最终dict_univ_bin的形式为:{文档数量x:[对应qid在list_num_docs中的索引],...}
        for ind, univ in enumerate(list_num_docs):
            if univ in dict_univ_bin:
                dict_univ_bin[univ].append(ind)
            else:
                bin = [ind]
                dict_univ_bin[univ] = bin
        #这里通过pre_allocate_batch得到的list_batch_inds是具有统一文档数量，被打包在一起的qid索引
        self.list_batch_inds = pre_allocate_batch(dict_univ_bin=dict_univ_bin, num_docs_per_batch=rough_batch_size)

    #定义了LETORSampler的迭代含义，那就是返回self.list_batch_inds中的所有索引
    def __iter__(self):
        for batch_inds in self.list_batch_inds:
            yield batch_inds

def load_data(data_id, data_dict,file,batch_size):
    _ltr_data = LTRDataset(data_id=data_id, data_dict=data_dict,file=file,batch_size=batch_size)
    letor_sampler = LETORSampler(data_source=_ltr_data, rough_batch_size=batch_size)
    #批训练，把数据变成一小批一小批数据进行训练。DataLoader就是用来包装所使用的数据，每次抛出一批数据
    #_ltr_data是数据来源，也就是数据集，而batch_sampler=letor_sampler定义了采样方法。
    ltr_data = torch.utils.data.DataLoader(_ltr_data, batch_sampler=letor_sampler, num_workers=0)
    #返回的是采样好的批数据
    return ltr_data

def load_multiple_data(data_id, dir_data, fold_k, batch_size=100):
    #dir_data是数据集的路径
    #fold_k是fold的index(下面的训练部分fold_k由flod_num即flod总数通过range函数和for循环得出)
    #最后的'/'是因为训练数据和测试数据都在flod文件夹内部
    fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
    #分别获得训练文件，交叉测试文件，测试文件
    file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'


    # _train_data = LTRDataset(data_id=data_id, data_dict=data_dict, file=file_train, batch_size=batch_size)
    # train_letor_sampler = LETORSampler(data_source=_train_data, rough_batch_size=batch_size)
    # train_data = torch.utils.data.DataLoader(_train_data, batch_sampler=train_letor_sampler, num_workers=0)
    train_data = load_data(data_id=data_id, data_dict=data_dict,file=file_train, batch_size=batch_size)

    # _test_data = LTRDataset(data_id=data_id, data_dict=data_dict, file=file_test, batch_size=batch_size)
    # test_letor_sampler = LETORSampler(data_source=_test_data, rough_batch_size=batch_size)
    # test_data = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)
    test_data = load_data(data_id=data_id, data_dict=data_dict,file=file_test,batch_size=batch_size)

    return train_data, test_data

data_dict = dict(data_id='MQ2008_Super', min_docs=1, min_rele=1,train_batch_size=100, validation_batch_size=100, test_batch_size=100)
#数据集的基本参数
data_id = 'MQ2008_Super'
batch_size = 50
file_train = 'D:/Data/MQ2008/Fold1/train.txt'
print(file_train)

#加载数据的测试
if __name__ == '__main__':
    train_data = load_data(data_id=data_id, data_dict=data_dict, file=file_train, batch_size=batch_size)
    print("train_data size", train_data.__sizeof__())
    for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:
        print("batch_ids", batch_ids)
        print("batch_q_doc_vectors.shape", batch_q_doc_vectors.shape)
        print("batch_std_labels", batch_std_labels.shape)