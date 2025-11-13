import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data

def DH_table(alpha, a, theta, d):
    """根据改进DH参数计算变换矩阵"""
    # TODO
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
        [0, 0, 0, 1]
    ])
    return np.matrix(T)

class MyCobotSimulator:
    def __init__(self, urdf_path, num_joints = 6):
        self.urdf_path = urdf_path
        self.num_joints = num_joints
        self.client = None
        self.robot_id = None
        self.plane_id = None
        # DH参数 - 按照(alpha, a, d)顺序
        self.dh_params = [
            (0, 0, 1.739),              # 关节1
            (np.pi/2, 0, 0),            # 关节2  
            (0, -1.35, 0),              # 关节3
            (0, -1.20, 0.8878),         # 关节4
            (np.pi/2, 0, 0.95),         # 关节5
            (-np.pi/2, 0, 0.655)        # 关节6
        ]
        self.offsets = [0, -np.pi/2, 0, -np.pi/2, 0, 0]  # 关节角度偏移
        self.fk = np.identity(4)   # 正向运动学结果矩阵

    def __enter__(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF(
            self.urdf_path,
            useFixedBase = True,
            flags = p.URDF_USE_SELF_COLLISION
        )
        self.plane_id = p.loadURDF("plane.urdf")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        p.disconnect(self.client)

    def forward_kinematics(self, joint_angles): 
        """正向运动学：返回末端变换矩阵"""
        # TODO
        # 初始化单位矩阵
        T_total = np.identity(4)
        
        # 计算每个关节的变换矩阵并相乘
        for i in range(self.num_joints):
            alpha, a, d = self.dh_params[i]  
            theta = joint_angles[i] + self.offsets[i]  
            
            # 计算当前关节的变换矩阵
            T_i = DH_table(alpha, a, theta, d)
            
            # 累积变换
            T_total = T_total * T_i
            
        self.fk = T_total
        return self.fk
    
    def get_pos(self):
        """
        获取位置 
        """
        # TODO: 实际实现末端位置计算
        pos = self.fk[0:3, 3].A1  # 提取位置向量
        return pos
    
    def get_rpy(self):
        """
        获取欧拉角 (roll, pitch, yaw)
        """
        # TODO: 实际实现欧拉角计算
        rotation = self.fk[0:3, 0:3]  # 提取旋转矩阵
        rpy = R.from_matrix(rotation).as_euler('xyz', degrees=False)
        return rpy

    def check_collision(self, joint_angles):
        """检查自碰撞和地面碰撞"""
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i])

        # 执行仿真以更新碰撞检测
        p.stepSimulation()

        # 自碰撞
        for i in range(self.num_joints):
            for j in range(i + 2, self.num_joints):
                if p.getContactPoints(self.robot_id, self.robot_id, i, j):
                    return True

        # 地面碰撞
        if p.getContactPoints(self.robot_id, self.plane_id):
            return True

        return False