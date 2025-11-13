import torch
from modules.modules import *
import ikpy.chain
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R
from utils.simulator import MyCobotSimulator
from scipy.interpolate import CubicHermiteSpline
import json
import time


def read_normalization_data(): # 读取归一化数据
    with open('data/normalization.json','r') as f:   #这里地址由data_generator函数保存
        normalization_data = json.load(f)
    pos_min = np.array(normalization_data["pos_min"])
    pos_max = np.array(normalization_data["pos_max"])
    rpy_min = np.array(normalization_data["rpy_min"])
    rpy_max = np.array(normalization_data["rpy_max"])
    return pos_min, pos_max, rpy_min, rpy_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    # 创建新的argparse命名空间并加载配置
    parser = argparse.ArgumentParser()
    for key, value in config_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser.parse_args([])  # 传入空列表避免解析命令行参数


def inference(cfg, upper, lower, hypernet, mainnet, sim, init_joint_angles = None, input_positions = None, delta = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float).to(device)):
    """
    进行推理计算，找到最小关节移动量对应的解
    :param cfg: 配置参数
    :param upper, lower: 关节角度的上下界
    :param hypernet: 超网络模型
    :param mainnet: 主网络模型
    :param input_positions: 输入位置数组
    :param init_joint_angles: 初始关节角度数组
    :return: 最小关节移动量对应的解
    """
    if input_positions is None:
        # 使用默认测试位置
        input_positions = np.array([[2, 1, 3, 3.14, 0.1, 0.2]])
        print("[WARNING] 使用默认测试位置")
        
    pos_min, pos_max, rpy_min, rpy_max = read_normalization_data()
    pos_norm = pos_max - pos_min
    rpy_norm = rpy_max - rpy_min

    # 将输入位置转换为torch张量并移动到GPU
    input_norm = np.zeros_like(input_positions)
    input_norm[0, :3] = (input_positions[0, :3] - pos_min) / pos_norm
    input_norm[0, 3:] = (input_positions[0, 3:] - rpy_min) / rpy_norm
    positions = torch.from_numpy(input_norm).float().to(device)
    # 初始化关节角度张量
    joint_angles = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float).to(device)

    predicted_weights = hypernet(positions)

    min_error = float('inf')
    min_pos_error = float('inf')
    min_rpy_error = float('inf')
    best_solution = None

    sampled_solutions = []
    if init_joint_angles is not None:
        init_joint_angles = init_joint_angles + 0.5 * delta
    else:
        init_joint_angles = 0.5 * delta
    for _ in range(cfg.num_solutions_validation):
        start = time.time()
        sample, _ = mainnet.validate(torch.ones(joint_angles.shape[0], 1).to(device),
                                          predicted_weights, lower, upper, init_joint_angles)
        end = time.time()
        sampled_solutions.append(sample)

    for sample in sampled_solutions:
        joint_angle = [sample[i][0].item() for i in range(cfg.num_joints)]
        print(f"关节角度: {np.round(np.rad2deg(joint_angle), 2)}")

        try:
            if sim.check_collision(joint_angle):
                print("检测到碰撞，跳过该解")
                continue

            # TODO
            # 计算正运动学得到末端位置
            fk_result = sim.forward_kinematics(joint_angle)
            fk_pos = fk_result[:3, 3].A1
            # 计算姿态
            rot_matrix = fk_result[:3, :3] 
            fk_rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)

            # 计算位置误差
            pos_error = np.linalg.norm(fk_pos - input_positions[0, :3])
            # 计算姿态误差
            rpy_error = np.linalg.norm(fk_rpy - input_positions[0, 3:])
            print(f"推理位姿：{fk_pos, fk_rpy}")
            print(f"位置测试损失 (RMSE): {pos_error}")
            print(f"姿态测试损失 (RMSE)(°): {np.rad2deg(rpy_error)}")
            # 更新最小误差及其对应的解
            if (pos_error + rpy_error) < min_error:
                min_error = pos_error + rpy_error
                min_pos_error = pos_error
                min_rpy_error = rpy_error
                best_solution = joint_angle
        except Exception as e:
            print(f"正运动学计算出错: {e}")

    print(f"最小位置误差当前为 (RMSE): {min_pos_error}")
    print(f"最小姿态误差当前为 (RMSE)(°): {np.rad2deg(min_rpy_error)}")
    print(f"对应的解为: {np.round(np.rad2deg(best_solution), 2)}")
    print("推理时间: {:.4f} 秒".format(end - start))
    return best_solution, min_error


if __name__ == '__main__':
    cfg = load_config("runs/exp_1/run_args.json")  #TODO: 自己训练保存的内容
    r_arm = ikpy.chain.Chain.from_urdf_file(cfg.chain_path)
        
    upper_bounds = [link.bounds[1] for link in r_arm.links[1:-1]]  # 上界
    lower_bounds = [link.bounds[0] for link in r_arm.links[1:-1]]  # 下界

    upper = torch.tensor(upper_bounds, dtype=torch.float32, device=device)
    lower = torch.tensor(lower_bounds, dtype=torch.float32, device=device)

    hypernet = HyperNet(cfg).to(device)
    mainnet = MainNet(cfg).to(device)

    hypernet.load_state_dict(torch.load('runs/exp_1/best_model.pt')) #TODO: 自己训练保存的内容
    hypernet.eval()

    with MyCobotSimulator(cfg.chain_path) as sim:
        solution = inference(cfg, upper, lower, hypernet, mainnet, sim)

