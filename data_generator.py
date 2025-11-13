import numpy as np
import h5py
import json
import ikpy.chain
from utils.simulator import MyCobotSimulator

def generate_dataset(sim: MyCobotSimulator, output_path, num_samples):
    r_arm = ikpy.chain.Chain.from_urdf_file(sim.urdf_path)

    # 利用urdf文件获取关节上下限
    upper = np.array([link.bounds[1] for link in r_arm.links[1:-1]])
    lower = np.array([link.bounds[0] for link in r_arm.links[1:-1]])
    
    input = []
    output = []
    number = 0

    while number < num_samples:
        joint_angles = (upper - lower) * np.random.rand(len(upper)) + lower
        if sim.check_collision(joint_angles):                    #自碰撞检测
            continue

        T = sim.forward_kinematics(joint_angles)                  #在sim里实现正运动学和获取位姿
        pose = np.concatenate([sim.get_pos(), sim.get_rpy()])

        input.append(joint_angles)
        output.append(pose)
        number += 1

    input = np.array(input)
    output = np.array(output)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('inputs', data = input)
        f.create_dataset('results', data = output)

    return output

def save_normalization_data(results, output_json_path):# TODO: 记住归一化数据保存位置
    # 保存归一化数据，这里保存的归一化数据是为了后续同时计算位置和姿态的误差
    results = np.array(results)
    normalization_data = {
        "pos_min": results[:, :3].min(axis=0).tolist(),
        "pos_max": results[:, :3].max(axis=0).tolist(),
        "rpy_min": results[:, 3:].min(axis=0).tolist(),
        "rpy_max": results[:, 3:].max(axis=0).tolist(),
    }
    with open(output_json_path, 'w') as f:
        json.dump(normalization_data, f, indent=4)

if __name__ == "__main__":
    URDF_PATH = "arm_files/new_mycobot_pro_320_pi_2022.urdf"     # TODO

    TRAIN_SAMPLES = 1280000    # TODO
    TEST_SAMPLES = 2560        # TODO

    TRAIN_PATH = f"data/train_{TRAIN_SAMPLES}.hdf5"  # TODO
    TEST_PATH = f"data/test_{TEST_SAMPLES}.hdf5"     # TODO 数据保存地址，保存格式为.hdf5文件

    with MyCobotSimulator(URDF_PATH) as sim:
        train_results = generate_dataset(sim, TRAIN_PATH, TRAIN_SAMPLES)
        test_results = generate_dataset(sim, TEST_PATH, TEST_SAMPLES)

        save_normalization_data(train_results, "data/normalization.json")
        print("✅ 数据生成完成")
