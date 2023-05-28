# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

'''
卡尔曼滤波分为两个阶段：
(1) 预测track在下一时刻的位置，
(2) 基于detection来更新预测的位置。
'''
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    对于每个轨迹，由一个 KalmanFilter 预测状态分布。每个轨迹记录自己的均值和方差作为滤波器输入。

    8维状态空间[x, y, a, h, vx, vy, va, vh]包含边界框中心位置(x, y)，纵横比a，高度h和它们各自的速度。
    物体运动遵循恒速模型。 边界框位置(x, y, a, h)被视为状态空间的直接观察（线性观察模型）

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        # 依据当前状态估计（高度）选择运动和观测不确定性。这些权重控制模型中的不确定性。
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        # Translates slice objects to concatenation along the first axis
        mean = np.r_[mean_pos, mean_vel]

        # 由测量初始化均值向量（8维）和协方差矩阵（8x8维）
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        #卡尔曼滤波器由目标上一时刻的均值和协方差进行预测。
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
       
        # 初始化噪声矩阵Q；np.r_ 按列连接两个矩阵
        # motion_cov是过程噪声 W_k的 协方差矩阵Qk 
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Update time state x' = Fx (1)
        # x为track在t-1时刻的均值，F称为状态转移矩阵，该公式预测t时刻的x'
        # self._motion_mat为F_k是作用在 x_{k-1}上的状态变换模型
        mean = np.dot(self._motion_mat, mean)
        # Calculate error covariance P' = FPF^T+Q (2)
        # P为track在t-1时刻的协方差，Q为系统的噪声矩阵，代表整个系统的可靠程度，一般初始化为很小的值，
        # 该公式预测t时刻的P'
        # covariance为P_{k|k} ，后验估计误差协方差矩阵，度量估计值的精确程度
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        投影状态分布到测量空间

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

       mean：ndarray，状态的平均向量（8维数组）。
       covariance：ndarray，状态的协方差矩阵（8x8维）。

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

       返回（ndarray，ndarray），返回给定状态估计的投影平均值和协方差矩阵

        """
        # 在公式4中，R为检测器的噪声矩阵，它是一个4x4的对角矩阵，
        # 对角线上的值分别为中心点两个坐标以及宽高的噪声，
        # 以任意值初始化，一般设置宽高的噪声大于中心点的噪声，
        # 该公式先将协方差矩阵P'映射到检测空间，然后再加上噪声矩阵R；
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
            
        # R为测量过程中噪声的协方差；初始化噪声矩阵R
        innovation_cov = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即 Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即 HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov # 公式(4)

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        通过估计值和观测值估计最新结果

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 将均值和协方差映射到检测空间，得到 Hx'和S
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K；相当于求解公式(5)
        # 公式5计算卡尔曼增益K，卡尔曼增益用于估计误差的重要程度
        # 求解卡尔曼滤波增益K 用到了cholesky矩阵分解加快求解；
        # 公式5的右边有一个S的逆，如果S矩阵很大，S的逆求解消耗时间太大，
        # 所以代码中把公式两边同时乘上S，右边的S*S的逆变成了单位矩阵，转化成AX=B形式求解。
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx' (3)
        # 在公式3中，z为detection的均值向量，不包含速度变化值，即z=[cx, cy, r, h]，
        # H称为测量矩阵，它将track的均值向量x'映射到检测空间，该公式计算detection和track的均值误差
        innovation = measurement - projected_mean

        # 更新后的均值向量 x = x' + Ky (6)
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # 更新后的协方差矩阵 P = (I - KH)P' (7)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
            状态分布上的平均向量（8维）
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
            状态分布的协方差（8x8维）
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
            N 个测量的 N×4维矩阵，每个矩阵的格式为（x，y，a，h），其中（x，y）是边界框中心位置，宽高比和h高度。
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
             如果为True，则只计算盒子中心位置

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
       返回一个长度为N的数组，其中第i个元素包含（mean，covariance）和measurements [i]之间的平方Mahalanobis距离

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
