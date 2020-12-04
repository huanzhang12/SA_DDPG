# SA-DDPG: State-adversarial DDPG for robust deep reinforcement learning
 
This folder contains a reference implementation for State-Adversarial DDPG
(SA-DDPG). SA-DDPG includes a robust regularization term based on SA-MDP to
obtain a DDPG agent that is robust to noises on state observations, including
adversarial perturbations. More details of our algorithm can be found in our
paper:

*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Bo Li, Duane Boning,* and *Cho-Jui
Hsieh*, "**Robust Deep Reinforcement Learning against Adversarial Perturbations
on State Observations**". [**NeurIPS 2020
(Spotlight)**](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf)
(\*Equal contribution)

Our DDPG implementation is based on
[ShangtongZhang/DeepRL](https://github.com/ShangtongZhang/DeepRL).  We use the
[auto_lirpa](https://github.com/KaidiXu/auto_LiRPA) library for computing
convex relaxations of neural networks, which provides a wide of range of
possibilities for the convex relaxation based method, including forward and
backward mode bound analysis ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)),
and interval bound propagation (IBP).

If you are looking for robust *on-policy* actor-critic algorithms, please see
our [SA-PPO repository](https://github.com/huanzhang12/SA_PPO).  If you are
looking for robust DQN (e.g., agents for Atari games) please see our [SA-DQN
repository](https://github.com/chenhongge/SA_DQN). 

## SA-DDPG Demo

Adversarial attacks on state observations (e.g., position and velocity
measurements) can easily make an agent fail. Our SA-DDPG agents are more robust
against adversarial attacks, including our strong Robust Sarsa (RS) attack.
                                                                                                                                                                             
| ![ant_vanilla_ddpg_attack_189.gif](/assets/ant_vanilla_ddpg_attack_189.gif) | ![ant_saddpg_attack_2025.gif](/assets/ant_saddpg_attack_2025.gif) | ![hopper_vanilla_attack_661.gif](/assets/hopper_vanilla_attack_661.gif) | ![hopper_saddpg_attack_1563.gif](/assets/hopper_saddpg_attack_1563.gif) |                             
|:--:| :--:| :--:| :--:|          
| Ant *Vanilla DDPG* <br> reward under attack: **189** | Ant *SA-DDPG* <br> reward under attack: **2025** | Hopper *Vanilla PPO* <br> reward under attack: **661** | Hopper *SA-PPO* <br> reward under attack: **1563** |

Note that DDPG is a representative off-policy actor-critic algorithm but it is
relatively early. For better agent performance, our proposed robust regularizer
can also be directly applied to newer methods such as TD3 or SAC.

## Setup

First clone this repository and install dependences (it is recommend to install
python packages into a fresh virtualenv or conda environment):

```
git submodule update --init
pip install -r requirements.txt
```

Python 3.7+ is required. Note that you need to install MuJoCo 1.5 first to use
the Gym environments. See
[here](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements)
for instructions.

## Pretrained agents

We release pretrained agents for vanilla DDPG and our robust SA-DDPG (solved by
convex relaxations or SGLD) in the `models` directory. These agent models can be loaded
by changing the  `--config` option and `--path_prefix` option:

```bash
# Ant SA-DDPG (solved by convex relaxation)
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex
# Ant SA-DDPG (solved by SGLD)
python eval_ddpg.py --config config/Ant_robust_sgld.json --path_prefix models/sa-ddpg-sgld
# Ant vanilla DDPG
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg
```

To run adversarial attacks on these agents, we can set
`test_config:attack_params:enabled=true` for `eval_ddpg.py`. For example:

```bash
# Attack SA-DDPG (convex relaxation) agent
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex test_config:attack_params:enabled=true
# Attack vanilla DDPG agent
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true
```

We will discuss more details on how to run adversarial attacks including our
newly proposed strong attacks (Robust Sarsa attack and maximal action
difference attack) in [this section](#robustness-evaluation). The default attack
method is the critic attack.


We list the performance of our pretrained agents below (these are agents with
**median** robustness, see explanations below):

| Environment         | Evaluation       | Vanilla DDPG | SA-DDPG (SGLD) | SA-DDPG (convex) |
|---------------------|------------------|--------------|----------------|-----------------------------|
| Ant-v2              | No attack        | 1487         | 2186           | 2254                        |
|                     | Strongest attack | 142          | **2007**       | 1820                        |
| Walker2d-v2         | No attack        | 1870         | 3318           | 4540                        |
|                     | Strongest attack | 790          | 1210           | **1986**                    |
| Hopper-v2           | No attack        | 3302         | 3068           | 3128                        |
|                     | Strongest attack | 606          | **1609**       | 1202                        |
| Reacher-v2          | No attack        | -4.37        | -5.00          | -5.24                       |
|                     | Strongest attack | -27.87       | **-12.10**     | -12.44                      |
| InvertedPendulum-v2 | No attack        | 1000         | 1000           | 1000                        |
|                     | Strongest attack | 92           | 423            | **1000**                    |

We attack each agent with 5 different attacks (random attack, critic attack, MAD
attack, RS attack and RS+MAD attack). Here we report the **lowest** reward of
all 5 attacks in "Strongest attack" rows. Additionally, **we train each setting
11 times** and we report the agent with **median** robustness (we do not
cherry-pick the best results). This is important due to the potential large
training variance in RL.

## Training SA-DDPG agents


We prepared training configurations for every environment evaluated in our
paper. The training hyperparameters are mostly from
[ShangtongZhang/DeepRL](https://github.com/ShangtongZhang/DeepRL). All training
config files are located in the `config` folder, and training can be invoked
using the `train_ddpg.py` script:

```bash
# Train DDPG Ant environment. Change Ant_vanilla.json to other config files to run other environments.
python train_ddpg.py --config config/Ant_vanilla.json
# Train SA-DDPG Ant environment (by default, solved using convex relaxations)
python train_ddpg.py --config config/Ant_robust.json
# Train SA-DDPG Ant environment (solved by SGLD)
python train_ddpg.py --config config/Ant_robust_sgld.json
```


Each environment contains 3 configuration files, one with `_robust` suffix
(SA-DDPG convex), one with `_robust_sgld` suffix (SA-DDPG SGLD) and one with
`_vanilla` suffix (vanilla DDPG). By default the code uses GPU for training. If
you want to run training on CPUs you can set `CUDA_VISIBLE_DEVICES=-1`, for
example:

```bash
# Train SA-DDPG Ant environment.
CUDA_VISIBLE_DEVICES=-1 python train_ddpg.py --config config/Ant_robust.json
```

We use the [auto_lirpa](https://github.com/KaidiXu/auto_LiRPA) library which
provides a wide of range of possibilities for the convex relaxation based
method, including forward and backward mode bound analysis, and interval bound
propagation (IBP).  In our paper, we only used the
[IBP+backward](https://github.com/huanzhang12/CROWN-IBP) scheme, which is
efficient and stable, and we did not report results using other possible
relaxations as this is not our main focus. If you are interested in trying
other relaxation schemes, e.g., if you want to use the cheapest IBP methods (at
the cost of potential stability issue), you just need the extra parameter
`training_config:robust_params:beta_scheduler:start=0.0`:

```bash
# The below training should run faster than original due to the use of cheaper relaxations.
# However, you probably need to train more iterations to compensate the instability of IBP.
python train_ddpg.py --config config/Ant_robust.json training_config:robust_params:beta_scheduler:start=0.0
```

To run evaluation after training, you can use the `eval_ddpg.py` script:

```bash
python eval_ddpg.py --config config/Ant_robust.json
```

We will discuss how to run adversarial attacks to evaluate policy robustness in
the [Robustness Evaluation](#robustness-evaluation) section.


## Robustness Evaluation

To enable adversarial attack during evaluation, add
`test_config:attack_params:enabled=true` to command line for `eval_ddpg.py`.
Attack type can be changed either in the config file (look for `test_config ->
attack_params -> type` in the JSON file), or specify via the command line as an
override (e.g., `test_config:attack_params:type=\"critic\"`). The perturbation
magnitude (eps) is defined under section `test_config -> attack_params -> eps`
and can be set in command line via `test_config:attack_params:eps=X` (`X` is
the eps value).  Since each state can have different ranges (e.g., position and
velocity can be in different magnitudes), we normalize the perturbation `eps`
based on standard deviation of each state value.

We implemented random attack, critic based attack and our newly proposed Robust
Sarsa (RS) and maximal action difference (MAD) attacks, which are detailed
below.

### Maximal Action Difference (MAD) Attack

We propose a maximal action difference (MAD) attack where we attempt to
maximize the KL divergence between the original action and the perturbed
action. It can be invoked by setting `test_config:attack_params:type` to
`action`. For example:

```bash
# MAD attack on vanilla DDPG agent for Ant:
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true test_config:attack_params:type=\"action\"
# MAD attack on SA-DDPG agent for Ant:
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex test_config:attack_params:enabled=true test_config:attack_params:type=\"action\"
```

The reported mean reward over 50 episodes for vanilla DDPG policy should be
less than 200 (reward without attacks is around 1500).  In contrast, our SA-DDPG
trained agent is more resistant to MAD attack, achieving an average reward over
2000.

### Robust Sarsa (RS) Attack

In our Robust Sarsa attack, we first learn a *robust* value function for the
policy under evaluation. Then, we attack the policy using this robust value
function. To invoke the RS attack, simply set `test_config:attack_params:type=\"sarsa\"`.
For example:

```bash
# RS attack on vanilla DDPG Ant policy.
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true test_config:attack_params:type=\"sarsa\"
# RS attack on SA-DDPG Ant policy.
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex test_config:attack_params:enabled=true test_config:attack_params:type=\"sarsa\"
```

The reported mean reward over 50 episodes for the vanilla agent should be
around 200 to 300. In contrast, attacking our SA-DDPG robust agent can only
reduce its reward to over 2000, 10 times better than the vanilla agent.

The attack script will save the trained value function into the model directory
if it does not exist, so that it won't need to train it again when you attack
using the same set of parameters.  The Robust Sarsa training step is usually
reasonably fast.

To delete all existing trained value functions, use:

```
# Delete all existing Sarsa value functions to train new ones.
MODELS_FOLDER=models/ # Change this path to your --path_prefix
find $MODELS_FOLDER -name '*_sarsa_*' -delete
```

This will guarantee to retrain the robust value functions for attack.

For the same policy under evaluation, you can try to train a robust value
function multiple times, and choose the best attack (lowest reward) as the
final result.  Additionally, Robust Sarsa attack has two hyperparameters for
robustness regularization, `reg` and `eps` (corresponding to
`test_config:sarsa_params:sarsa_reg` and
`test_config:sarsa_params:action_eps_scheduler:end` in the configuration file),
to build the robust value function.  Although the default settings generally
work well, for a comprehensive robustness evaluation it is recommend to run
Robust Sarsa attack under different hyperparameters and choose the best attack
(lowest reward) as the final result. For example, you can do scanning in the 
following way:

```bash
MODEL_CONFIG=config/Ant_vanilla.json
MODELS_FOLDER=models/vanilla-ddpg # Change this path to your --path_prefix
export CUDA_VISIBLE_DEVICES=-1 # Not using GPU.
# First remove all existing value functions.
find $MODELS_FOLDER -name '*_sarsa_*' -delete
# Scan RS attack hyperparameters.
for sarsa_eps in 0.02 0.05 0.1 0.15 0.2 0.3; do
    for sarsa_reg in 0.1 0.3 1.0 3.0 10.0; do
        # We run all attacks in parallel (assuming we have sufficient CPUs); remove the last "&" if you want to run attacks one by one.
        python eval_ddpg.py --config ${MODEL_CONFIG} --path_prefix ${MODELS_FOLDER} test_config:attack_params:enabled=true test_config:attack_params:type=\"sarsa\" test_config:sarsa_params:sarsa_reg=${sarsa_reg} test_config:sarsa_params:action_eps_scheduler:end=${sarsa_eps} &
    done
done
wait
```

After this script finishes, you will find all attack logs in
`models/vanilla-ddpg/ddpg_Ant-v2/log/` (or the corresponding directory as you
specified in `MODEL_CONFIG` and `MODELS_FOLDER`):

```bash
# Read the average reward over all settings
tail -n 2 models/vanilla-ddpg/ddpg_Ant-v2/log/*eval*.txt
# Take the min of all reported "Average Reward"
```

### RS+MAD attack

We additionally provide a combined attack of RS+MAD, which can be invoked by
setting attack type to `sarsa_action`. It is usually stronger than RS or MAD
attack alone. For example:

```bash
# RS+MAD attack on vanilla DDPG Ant policy.
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true test_config:attack_params:type=\"sarsa_action\"
# RS+MAD attack on SA-DDPG Ant policy.
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex test_config:attack_params:enabled=true test_config:attack_params:type=\"sarsa_action\"
```

For the vanilla Ant agent, our RS+MAD attack achieves a very low reward of about 150
averaged over 50 episodes, which is only 1/10 of the original agent's
performance.  On the other hand, our SA-DDPG agent still has an average reward
of about 1800.

An additional parameter, `test_config:attack_params:sarsa_action_ratio` can be
set to a number between 0 and 1, to set the ratio between Robust Sarsa and MAD
attack loss. Usually, you first find the best Sarsa model (by scanning `reg`
and `eps` for RS attack as the script above), and then try to further enhance
the attack using a non-zero `sarsa_action_ratio`.


### Critic based attack and random attack

Critic based attack and random attack are two relatively weak baseline attacks.
They can be used by setting `test_config:attack_params:type` to `critic` and
`random`, respectively:

```bash
# Critic based attack on vanilla DDPG agent for Ant:
python eval_ddpg.py --config config/Ant_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true test_config:attack_params:type=\"critic\"
# Critic based attack on SA-DDPG agent for Ant:
python eval_ddpg.py --config config/Ant_robust.json --path_prefix models/sa-ddpg-convex test_config:attack_params:enabled=true test_config:attack_params:type=\"critic\"
```

