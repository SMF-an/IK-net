from turtle import pos
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch
from modules.modules import *
from utils.dataset import IKDataset, IKDatasetVal
import ikpy.chain
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
from utils.simulator import MyCobotSimulator

def read_normalization_data(): # 读取归一化数据
    with open('data/normalization.json','r') as f:   #这里地址由data_generator函数保存
        normalization_data = json.load(f)
    pos_min = np.array(normalization_data["pos_min"])
    pos_max = np.array(normalization_data["pos_max"])
    rpy_min = np.array(normalization_data["rpy_min"])
    rpy_max = np.array(normalization_data["rpy_max"])
    return pos_min, pos_max, rpy_min, rpy_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg):
    r_arm = ikpy.chain.Chain.from_urdf_file(cfg.chain_path)
    sim = MyCobotSimulator(cfg.chain_path)

    # 利用urdf文件获得关节上下界
    upper_bounds = [link.bounds[1] for link in r_arm.links[1:-1]]  # 上界
    lower_bounds = [link.bounds[0] for link in r_arm.links[1:-1]]  # 下界

    upper = torch.tensor(upper_bounds, dtype=torch.float32, device=device)
    lower = torch.tensor(lower_bounds, dtype=torch.float32, device=device)
    
    # 加载数据
    train_dataset = IKDataset(cfg.train_data_path)
    test_dataset = IKDatasetVal(cfg.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, 
                                  shuffle = True, num_workers=0, pin_memory = True)
    test_dataloader = DataLoader(test_dataset, batch_size = cfg.batch_size, 
                                 shuffle = False, num_workers = 0, pin_memory = True)
    
    hypernet = HyperNet(cfg).to(device)
    mainnet = MainNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        hypernet.parameters(),
        lr=cfg.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # loss数据初始化
    pos_min, pos_max, rpy_min, rpy_max = read_normalization_data()
    pos_norm = pos_max - pos_min
    rpy_norm = rpy_max - rpy_min
    
    train_counter = 0
    train_loss = 0.0

    best_test_loss = np.inf
    epochs_without_improvements = 0

    train_losses = []
    pos_losses = []
    ori_losses = []

    for epoch in range(cfg.num_epochs):
        hypernet.train()
        for positions, joint_angles in train_dataloader:
            positions = positions.to(device)
            joint_angles = joint_angles.to(device)
            output = torch.cat((torch.ones(joint_angles.shape[0], 1).to(device), joint_angles), dim = 1)

            optimizer.zero_grad()
            with autocast():
                predicted_weights = hypernet(positions)
                distributions, _ = mainnet(output, predicted_weights)
                losses = [-torch.mean(distributions[i].log_prob(joint_angles[:, i].unsqueeze(1))) 
                          for i in range(len(distributions))]
                # 负对数似然函数，给出已有分布，算对应点的概率密度的负对数
                loss = sum(losses)/len(losses)
                
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)  # 取消缩放以进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_counter += 1
            train_loss += loss.item()
            
        train_losses.append(train_loss / train_counter)
        print(f"Train loss (Likelihood) {train_losses[-1]}")
        train_loss, train_counter = 0, 0
        delta = np.zeros(6)
        delta_tensor = torch.from_numpy(delta).float().to(device)

        if epoch % 2 == 0:
            hypernet.eval()
            
            all_best_pos_errors = []
            all_best_ori_errors = []

            with torch.no_grad(): 
                for positions, joint_angles in test_dataloader:
                    positions = positions.to(device)
                    joint_angles = joint_angles.to(device)
                    predicted_weights = hypernet(positions)

                    batch_pos_errors_list = []
                    batch_ori_errors_list = []

                    for _ in range(cfg.num_solutions_validation):
                        sample, distributions = mainnet.validate(torch.ones(joint_angles.shape[0], 1).to(device),
                                                                  predicted_weights, lower, upper, delta_tensor)
                        
                        current_sample_pos_errors = []
                        current_sample_ori_errors = []

                        for k in range(len(positions)):
                            joint_angle = [sample[i][k].item() for i in range(cfg.num_joints)]
                            
                            # TODO
                            # 计算归一化前的位置均方根误差（RMSE）
                            # 计算归一化前的姿态均方根误差（RMSE）
                            # 计算归一化后的位置均方根误差（RMSE）
                            # 计算归一化后的姿态均方根误差（RMSE）
                            # 计算总误差（可按需调整权重等）
                            # 计算采样关节角度对应的实际末端位姿
                            fk_results = sim.forward_kinematics(joint_angle)
                            fk_pos = torch.tensor(fk_results[:3, 3].A1, dtype=torch.float32, device=device)
                            fk_ori_matrix = fk_results[:3, :3]
                            fk_rpy = torch.tensor(R.from_matrix(fk_ori_matrix).as_euler('xyz', degrees=False), dtype=torch.float32, device=device)
                            
                            fk_pos = (fk_pos - torch.from_numpy(pos_min).float().to(device)) / torch.from_numpy(pos_norm).float().to(device)
                            fk_rpy = (fk_rpy - torch.from_numpy(rpy_min).float().to(device)) / torch.from_numpy(rpy_norm).float().to(device)
                            
                            target_pos = positions[k, :3] 
                            target_ori = positions[k, 3:] 

                            pos_error_rmse = (torch.sum((fk_pos - target_pos) ** 2).item()) ** 0.5
                            ori_error_rmse = (torch.sum((fk_rpy - target_ori) ** 2).item()) ** 0.5
                            
                            current_sample_pos_errors.append(pos_error_rmse)
                            current_sample_ori_errors.append(ori_error_rmse)
                        
                        batch_pos_errors_list.append(current_sample_pos_errors)
                        batch_ori_errors_list.append(current_sample_ori_errors)
 
                    batch_pos_errors_tensor = torch.tensor(batch_pos_errors_list, device=device).T
                    batch_ori_errors_tensor = torch.tensor(batch_ori_errors_list, device=device).T

                    min_pos_errors, _ = torch.min(batch_pos_errors_tensor, dim=1)
                    min_ori_errors, _ = torch.min(batch_ori_errors_tensor, dim=1)

                    all_best_pos_errors.extend(min_pos_errors.cpu().tolist())
                    all_best_ori_errors.extend(min_ori_errors.cpu().tolist())

            final_pos_loss = np.mean(all_best_pos_errors)
            final_ori_loss = np.mean(all_best_ori_errors)
            final_test_loss = final_pos_loss + final_ori_loss 

            pos_losses.append(final_pos_loss)
            ori_losses.append(final_ori_loss)
            
            print(f"Position test loss (Mean of Min RMSE): {pos_losses[-1]}")
            print(f"Orientation test loss (Mean of Min RMSE): {ori_losses[-1]}")
            print()

            plt.plot(range(len(train_losses)), train_losses, label = 'train')
            plt.savefig(f'{cfg.exp_dir}/train_plot.png')
            plt.clf()
            plt.plot(range(len(pos_losses)), pos_losses, label = 'pos_test')
            plt.savefig(f'{cfg.exp_dir}/pos_test_plot.png')
            plt.clf()
            plt.plot(range(len(ori_losses )), ori_losses, label = 'rpy_test')
            plt.savefig(f'{cfg.exp_dir}/rpy_test_plot.png')
            plt.clf()

            torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/last_model.pt')
            torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/last_optimizer.pt')

            if final_test_loss < best_test_loss:
                #TODO 实现模型更新和早停止(best_test_loss, epochs_without_improvements)
                best_test_loss = final_test_loss
                epochs_without_improvements = 0
                torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/best_model.pt')
                torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/best_optimizer.pt')
                with open(f'{cfg.exp_dir}/best_test_loss.txt', 'a+') as f:
                    f.write(f'Epoch {epoch} - pos test loss: {pos_losses[-1]}, rpy test loss: {ori_losses[-1]} \n')
            else:
                #TODO 实现训练早停止，早停止参数为cfg.early_stopping_epochs
                epochs_without_improvements += 2
                if epochs_without_improvements >= cfg.early_stopping_epochs:
                    print(f"Early stopping triggered after {epochs_without_improvements} epochs without improvement.")
                    break

def create_exp_dir(cfg):
    if not os.path.exists(cfg.exp_dir):
        os.mkdir(cfg.exp_dir)
    existing_dirs = os.listdir(cfg.exp_dir)
    if existing_dirs:
        sorted_dirs = sorted(existing_dirs, key=lambda x : int(x.split('_')[1]))
        last_exp_num = int(sorted_dirs[-1].split('_')[1])
        exp_name = f"{cfg.exp_dir}/exp_{last_exp_num + 1}"
    else:
        exp_name = f"{cfg.exp_dir}/exp_0"
    os.makedirs(exp_name)
    with open(f'{exp_name}/run_args.json', 'w+') as f:
        json.dump(cfg.__dict__, f, indent=2)
    return exp_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain-path', type=str, default="arm_files/new_mycobot_pro_320_pi_2022.urdf", help='urdf chain path')     #TODO
    parser.add_argument('--train-data-path', type=str, default="data/train_1280000.hdf5", help='train data path')       #TODO
    parser.add_argument('--test-data-path', type=str, default='data/test_2560.hdf5', help='test data path')           #TODO
    parser.add_argument('--num-joints', type=int, default=6, help='number of joints of the kinematic chain')                #TODO
    
    #以下为可以修改优化的超参数
    parser.add_argument('--early-stopping-epochs', type=int, default=20, help='number of epochs without improvement to trigger end of training') 
    parser.add_argument('--hypernet-input-dim', type=int, default=6, help='number of input to the hypernetwork (f)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--num-solutions-validation', type=int, default=5, help='solutions number')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--embedding-dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--hypernet-hidden-size', type=int, default=512, help='hypernetwork (f) number of neurons in hidden layer')
    parser.add_argument('--hypernet-num-hidden-layers', type=int, default=4, help='hypernetwork  (f) number of hidden layers')
    parser.add_argument('--jointnet-hidden-size', type=int, default=256, help='jointnet (g) number of neurons in hidden layer')
    parser.add_argument('--jointnet-num-layers', type=int, default=3, help='Total layers in jointnet (g) (e.g., 3 for Input->H1->H2->Out)')
    parser.add_argument('--num-gaussians', type=int, default=10, help='number of gaussians for mixture . default=1 no mixture')
    parser.add_argument('--encoder-hidden-dim', type=int, default=64, help='hidden dim for pos/rpy encoders')
    parser.add_argument('--encoder-embedding-dim', type=int, default=64, help='output dim for pos/rpy encoders')
    
    parser.add_argument('--with-orientation', action='store_true', default=True, help='Whether to include orientation information')
    parser.add_argument('--grad-clip', type=int, default=1, help='clip norm of gradient')
    parser.add_argument('--exp_dir', type=str, default='runs', help='folder path name to save the experiment')

    parser.set_defaults()
    cfg = parser.parse_args()

    cfg.jointnet_output_dim = cfg.num_gaussians * 2 + cfg.num_gaussians if cfg.num_gaussians != 1 else 2

    full_exp_dir = create_exp_dir(cfg) #这里已经将cfg文件保存，用于后续直接调用

    cfg.exp_dir = full_exp_dir

    train(cfg)
