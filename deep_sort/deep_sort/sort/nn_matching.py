# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    
    用于计算成对点之间的平方距离
    a ：NxM 矩阵，代表 N 个样本，每个样本 M 个数值 
    b ：LxM 矩阵，代表 L 个样本，每个样本有 M 个数值 
    返回的是 NxL 的矩阵，比如 dist[i][j] 代表 a[i] 和 b[j] 之间的平方和距离
    参考：https://blog.csdn.net/frankzd/article/details/80251042

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    用于计算成对点之间的余弦距离
    a ：NxM 矩阵，代表 N 个样本，每个样本 M 个数值 
    b ：LxM 矩阵，代表 L 个样本，每个样本有 M 个数值 
    返回的是 NxL 的矩阵，比如 c[i][j] 代表 a[i] 和 b[j] 之间的余弦距离
    参考：
    https://blog.csdn.net/u013749540/article/details/51813922
    

    """
    if not data_is_normalized:
        # np.linalg.norm 求向量的范式，默认是 L2 范式 
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T) # 余弦距离 = 1 - 余弦相似度


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    对于每个目标，返回最近邻居的距离度量, 即与到目前为止已观察到的任何样本的最接近距离。

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
        匹配阈值。 距离较大的样本对被认为是无效的匹配。
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
        如果不是None，则将每个类别的样本最多固定为该数字。 
        删除达到budget时最古老的样本。

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
        一个从目标ID映射到到目前为止已经观察到的样本列表的字典

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance # 欧式距离
        elif metric == "cosine":
            self._metric = _nn_cosine_distance # 余弦距离
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget # budge用于控制 feature 的数目
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        用新的数据更新测量距离

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        传入特征列表及其对应id，partial_fit构造一个活跃目标的特征字典。

        """
        for feature, target in zip(features, targets):
            # 对应目标下添加新的feature，更新feature集合
            # samples字典    d: feature list}
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                # 只考虑budget个目标，超过直接忽略
                self.samples[target] = self.samples[target][-self.budget:]
        
        # 筛选激活的目标；samples是一个字典{id->feature list}
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        
        计算features和targets之间的距离，返回一个成本矩阵（代价矩阵）
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
