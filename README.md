## IK-net

This repository contains code for learning inverse kinematics with a hypernetwork-based approach. The project includes utilities to generate a dataset from a robot URDF, train the model, and run inference to compute joint-angle solutions for target end-effector poses.

#### Environment Setup

- **Python:** 3.9+ recommended.
- **PyTorch:** >=2.0
- **Packages:** Create the environment from `environment.yml`.

Shell example to create the environment (using conda):

```shell
conda env create -f environment.yml -n iknet
conda activate iknet
```

If you don't use conda, install dependencies listed in `environment.yml` with your preferred tool. Key packages include `torch`, `pybullet`, `ikpy`, `scipy`, `h5py`, `numpy` and `matplotlib`.

#### Project layout (important files)

- `data_generator.py`: sample joint space, check collisions, save `.hdf5` datasets and `data/normalization.json`.
- `train.py`: training script that reads dataset and trains the hypernetwork/main network.
- `inference.py`: example script to run inference using a saved experiment config and model weights.
- `arm_files/`: put your robot URDF and any required meshes here.
- `utils/simulator.py`: PyBullet-based simulator used for forward kinematics and collision checking.

#### 1) Prepare your robot URDF

- Put your robot URDF and any referenced mesh files inside the `arm_files/` folder.
- Ensure the URDF references meshes using relative paths that are valid from the repository root (or update the URDF accordingly).

##### Important: Update DH parameters for your robot

- The repository uses a simple DH-based forward kinematics implementation inside `utils/simulator.py`. Before generating data or running training you must verify (and if necessary update) the DH parameter table and related settings to match your robot.
- Open `utils/simulator.py` and locate the `MyCobotSimulator` class. The following fields are important:
	- `dh_params`: a list of tuples `(alpha, a, d)` describing the modified DH parameters for each revolute joint in order.
	- `offsets`: a list of per-joint angle offsets (radians) applied to joint values before FK.
	- `num_joints`: the number of joints the simulator will iterate over (default is `6` in this project).

Example excerpt (format used in this codebase):

```python
# inside utils/simulator.py
self.dh_params = [
		(0, 0, 1.739),
		(np.pi/2, 0, 0),
		(0, -1.35, 0),
		(0, -1.20, 0.8878),
		(np.pi/2, 0, 0.95),
		(-np.pi/2, 0, 0.655)
]
self.offsets = [0, -np.pi/2, 0, -np.pi/2, 0, 0]
```

#### 2) Generate a dataset (URDF -> .hdf5)

- Edit `data_generator.py` to point to your URDF. The default variable is `URDF_PATH` in the file or you can modify and run like below.
- The script saves datasets to `data/train_<N>.hdf5` and `data/test_<M>.hdf5` and writes normalization ranges to `data/normalization.json`.
- If you need fewer samples for quick tests, edit `TRAIN_SAMPLES` and `TEST_SAMPLES` constants at the bottom of `data_generator.py`.

Notes:
- The generator uses `PyBullet` (DIRECT mode) and the simulator's `check_collision` to filter out self-colliding poses. If your URDF has different joint ordering or additional joints, you may need to adapt `MyCobotSimulator` and `data_generator.py`.

#### 3) Train the model

- Training reads the `.hdf5` files and the `data/normalization.json` file. The default paths are set in `train.py` CLI arguments.

PowerShell example (basic run):

```shell
# run training with defaults (edit args to point to your files if needed)
python train.py --chain-path arm_files/my_robot.urdf --train-data-path data/train_1280000.hdf5 --test-data-path data/test_2560.hdf5 --num-epochs 100 --batch-size 512 --exp_dir runs
```

Tips:
- The script creates an experiment folder under `runs/` (e.g., `runs/exp_0`) and saves `run_args.json`, model checkpoints and plots there.
- For faster training use a machine with CUDA and confirm `torch.cuda.is_available()`.

#### 4) Inference (use a trained model)

- `inference.py` demonstrates loading the saved `run_args.json` and a `best_model.pt` checkpoint. You can edit the code to point to your experiment directory and weights.

To run inference on a specific pose programmatically, modify `input_positions` in `inference.py` or adapt `inference()` to accept a file or CLI arguments.

#### 5) Common adjustments & troubleshooting

- URDF issues: If `PyBullet` fails to load your URDF, check mesh paths and units. Use `pybullet.GUI` (replace `p.connect(p.DIRECT)` with `p.connect(p.GUI)`) to visually debug.
- Joint ordering: `ikpy` extracts joint bounds from the URDF. If your robot uses fixed joints or a different chain definition, make sure `ikpy.chain.Chain.from_urdf_file()` returns the expected links and `utils/simulator.py` expects the same joint count.
- Debugging collisions: If many valid samples are rejected, visualize a few joint configurations in GUI mode to confirm contact timings.

#### 6) Example quick workflow

- Place URDF: `arm_files/my_robot.urdf`
- Generate small dataset (for quick iteration): edit `TRAIN_SAMPLES = 20000` and `TEST_SAMPLES = 2000` in `data_generator.py`, then run `python data_generator.py`.
- Train for a short test: `python train.py --chain-path arm_files/my_robot.urdf --train-data-path data/train_20000.hdf5 --test-data-path data/test_2000.hdf5 --num-epochs 10 --batch-size 256 --exp_dir runs`
- Use the produced `runs/exp_X/run_args.json` and `runs/exp_X/best_model.pt` with `inference.py`.


Reference paper: [Neural Inverse Kinematics](https://arxiv.org/abs/2205.10837)