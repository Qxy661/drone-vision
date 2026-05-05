"""
卡尔曼滤波目标跟踪器
Kalman Filter Target Tracker

卡尔曼滤波是最优状态估计器, 在目标跟踪中用于:
1. 平滑检测噪声: 检测结果有抖动, 卡尔曼滤波输出平滑轨迹
2. 预测遮挡位置: 目标被遮挡时, 用运动模型预测位置
3. 速度估计: 从位置序列估计目标速度, 用于预测

状态向量: [x, y, vx, vy] (位置 + 速度)
观测向量: [x, y] (检测到的位置)

运动模型: 匀速运动 (Constant Velocity)
  x[k+1] = x[k] + vx[k] * dt
  y[k+1] = y[k] + vy[k] * dt

优势:
- 实时性好: O(n^3) 矩阵运算, n=4 很小
- 鲁棒性强: 能处理检测丢失、噪声、误检
- 工业界标准: SORT/DeepSORT 的核心组件
"""
import numpy as np
from typing import Optional, Tuple


class KalmanTracker:
    """2D 卡尔曼滤波跟踪器

    状态: [x, y, vx, vy]
    观测: [x, y]

    使用方法:
        tracker = KalmanTracker(dt=0.033)  # 30fps
        tracker.init(x=320, y=240)
        while True:
            tracker.predict()
            if detection:
                tracker.update(det_x, det_y)
            estimated_pos = tracker.get_state()
    """
    def __init__(self, dt: float = 0.033,
                 process_noise: float = 1.0,
                 measurement_noise: float = 10.0):
        self.dt = dt

        # 状态转移矩阵 F (匀速模型)
        # [x]   [1 0 dt 0] [x]
        # [y] = [0 1 0 dt] [y]
        # [vx]  [0 0 1  0] [vx]
        # [vy]  [0 0 0  1] [vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # 观测矩阵 H (只观测位置)
        # [z_x]   [1 0 0 0] [x]
        # [z_y] = [0 1 0 0] [y]
        #                        [vx]
        #                        [vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # 过程噪声协方差 Q
        q = process_noise
        self.Q = np.eye(4) * q
        self.Q[0, 0] = q * dt * dt / 2
        self.Q[1, 1] = q * dt * dt / 2
        self.Q[2, 2] = q * dt
        self.Q[3, 3] = q * dt

        # 测量噪声协方差 R
        self.R = np.eye(2) * measurement_noise

        # 状态和协方差
        self.x = np.zeros(4)  # 状态向量
        self.P = np.eye(4) * 100  # 初始协方差 (大 = 不确定)
        self.initialized = False

        # 跟踪质量
        self.hits = 0        # 连续匹配次数
        self.misses = 0      # 连续丢失次数
        self.age = 0         # 总帧数

    def init(self, x: float, y: float):
        """用首次检测初始化状态"""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 100
        self.P[2, 2] = self.P[3, 3] = 25  # 速度不确定
        self.initialized = True
        self.hits = 1
        self.misses = 0
        self.age = 1

    def predict(self) -> Tuple[float, float]:
        """预测步骤: 用运动模型推算下一时刻状态"""
        if not self.initialized:
            return (0.0, 0.0)

        # 状态预测: x' = F * x
        self.x = self.F @ self.x

        # 协方差预测: P' = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        return (self.x[0], self.x[1])

    def update(self, z_x: float, z_y: float):
        """更新步骤: 用观测修正预测

        卡尔曼增益: K = P * H^T * (H * P * H^T + R)^-1
        状态更新: x = x + K * (z - H * x)
        协方差更新: P = (I - K * H) * P
        """
        if not self.initialized:
            self.init(z_x, z_y)
            return

        z = np.array([z_x, z_y])

        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新
        y = z - self.H @ self.x  # 残差 (innovation)
        self.x = self.x + K @ y

        # 协方差更新 (Joseph form, 数值更稳定)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.hits += 1
        self.misses = 0

    def mark_missed(self):
        """标记本帧未检测到目标"""
        self.misses += 1
        self.hits = 0

    def get_state(self) -> Tuple[float, float, float, float]:
        """返回当前状态估计: (x, y, vx, vy)"""
        return tuple(self.x)

    def get_position(self) -> Tuple[float, float]:
        return (self.x[0], self.x[1])

    def get_velocity(self) -> Tuple[float, float]:
        return (self.x[2], self.x[3])

    def get_position_uncertainty(self) -> float:
        """位置不确定度 (P矩阵的位置部分迹)"""
        return float(np.trace(self.P[:2, :2]))

    @property
    def is_confirmed(self) -> bool:
        """跟踪确认: 连续匹配 >= 3 次"""
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        """跟踪丢失: 连续丢失 >= 10 帧"""
        return self.misses >= 10


class MultiTargetTracker:
    """多目标跟踪器 (简化版 SORT)

    管理多个 KalmanTracker 实例
    用 IoU (Intersection over Union) 匹配检测和跟踪

    工作流程:
    1. 对已有跟踪器做 predict
    2. 用 IoU 匹配新检测和已有跟踪
    3. 匹配成功 -> update, 未匹配 -> mark_missed
    4. 新检测 -> 创建新跟踪器
    5. 丢失太久 -> 删除跟踪器
    """
    def __init__(self, dt: float = 0.033, iou_threshold: float = 0.3,
                 max_missed: int = 10):
        self.trackers: list[KalmanTracker] = []
        self.dt = dt
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0

    def update(self, detections: list) -> list:
        """输入检测列表, 输出跟踪结果

        Args:
            detections: [(cx, cy, w, h), ...] 检测框

        Returns:
            [(track_id, cx, cy, vx, vy, confidence), ...]
        """
        # Step 1: Predict all trackers
        for t in self.trackers:
            t.predict()

        # Step 2: Match detections to trackers (simple greedy IoU)
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # Step 3: Update matched trackers
        for det_idx, trk_idx in matched:
            cx, cy, w, h = detections[det_idx]
            self.trackers[trk_idx].update(cx, cy)

        # Step 4: Mark unmatched trackers as missed
        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].mark_missed()

        # Step 5: Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            cx, cy, w, h = detections[det_idx]
            t = KalmanTracker(dt=self.dt)
            t.init(cx, cy)
            t.track_id = self.next_id
            self.next_id += 1
            self.trackers.append(t)

        # Step 6: Remove dead trackers
        self.trackers = [
            t for t in self.trackers if t.misses < self.max_missed
        ]

        # Return confirmed tracks
        results = []
        for t in self.trackers:
            if t.is_confirmed:
                x, y, vx, vy = t.get_state()
                results.append({
                    'track_id': t.track_id,
                    'x': x, 'y': y,
                    'vx': vx, 'vy': vy,
                    'hits': t.hits,
                    'age': t.age,
                })
        return results

    def _match(self, detections):
        """简单贪心匹配 (基于距离, 非IoU)"""
        if not self.trackers or not detections:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        # 计算距离矩阵
        cost_matrix = np.zeros((len(detections), len(self.trackers)))
        for i, (cx, cy, w, h) in enumerate(detections):
            for j, t in enumerate(self.trackers):
                tx, ty = t.get_position()
                cost_matrix[i, j] = np.sqrt((cx - tx)**2 + (cy - ty)**2)

        # 贪心匹配
        matched = []
        used_dets = set()
        used_trks = set()

        while True:
            if cost_matrix.size == 0:
                break
            min_idx = np.unravel_index(
                np.argmin(cost_matrix), cost_matrix.shape)
            min_cost = cost_matrix[min_idx]

            if min_cost > 200:  # 距离阈值 (像素)
                break

            i, j = min_idx
            if i not in used_dets and j not in used_trks:
                matched.append((i, j))
                used_dets.add(i)
                used_trks.add(j)

            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [j for j in range(len(self.trackers)) if j not in used_trks]

        return matched, unmatched_dets, unmatched_trks
