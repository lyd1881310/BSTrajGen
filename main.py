import yaml
from argparse import ArgumentParser

from train import train_act_gen
from generate import run_act_generate, run_time_generate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='fsq_global')
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(open('configs/act_gen_config.yaml', 'r'))
    cfg.update(vars(args))

    train_act_gen(cfg)
    run_act_generate(cfg)
    run_time_generate(exp_id=cfg['exp_id'])
