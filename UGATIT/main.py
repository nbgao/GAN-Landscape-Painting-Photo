# nohup python -u main.py --light=True --iteration=1000000 --ch=32 --device=cuda:0 > log_run_landscape.txt 2>&1 &
# python main.py --light=True --iteration=10000 --ch=16 --device=cuda:3         1897M
# python main.py --light=True --iteration=10000 --ch=32 --device=cuda:1         3791M
# python main.py --light=False --iteration=10000 --ch=32 --device=cuda:1        6765M -> 8638M

# nohup python -u main.py --ch=32 --device=cuda:1 > log_run_1.txt 2>&1 &
# nohup python -u main.py --light=True --img_size=128 --device=cuda:0 > log_run_2.txt 2>&1 &   8701M ->
# nohup python -u main.py --dataset=Landscape1 --light=True --img_size=256 --device=cuda:2 > log_run_landscape_3.txt 2>&1 &
# nohup python -u main.py --dataset=Landscape1 --light=True --img_size=256 --device=cuda:2 --resume=True > log_run_landscape_3_resume.txt 2>&1 &

# python main.py --phase=test --dataset=Landscape --light=False --ch=32 --img_size=256 --device=cuda:0 --test_iter=700000
# python main.py --phase=test --dataset=Landscape1 --light=True --ch=64 --img_size=256 --device=cuda:1 --test_iter=900000

from UGATIT import UGATIT
import argparse
from util import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='Landscape1', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda:3', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--test_iter', type=int, default=600000, help='Test load model iteration')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        # gan.test()
        gan.test2()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
