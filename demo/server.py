'''
@Filename: server.py
@Author: nbgao (Gao Pengbing)
@Contact: nbgao@126.com
'''
import argparse
import time, itertools
import io
import os
import sys
sys.path.append('..')
from dataset import ImageFolder
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob

import flask

# python main.py --phase=test --light=True --ch=32 --img_size=256 --device=cuda:1 --test_iter=900000

app = flask.Flask(__name__)
gan = None

def image_loader(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    img = img.convert('RGB')
    img = gan.test_transform(img)
    img = img.unsqueeze(0)
    return img


def A2B_infer(image_buffer_A):
    real_A = image_loader(image_buffer_A)
    gan.genA2B.eval()

    real_A = real_A.to(gan.device)
    fake_A2B, _, fake_A2B_heatmap = gan.genA2B(real_A)
    # fake_A2B2A, _, fake_A2B2A_heatmap = gan.genB2A(fake_A2B)
    # fake_A2A, _, fake_A2A_heatmap = gan.genB2A(real_A)
    A2B = 255.0 * tensor2numpy(denorm(fake_A2B[0]))
    
    cv2.imwrite(os.path.join('result', 'A2B', 'A2B_{}.png'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime()))), RGB2BGR(A2B))
    return A2B


def B2A_infer(image_buffer_B):
    real_B = image_loader(image_buffer_B)
    gan.genB2A.eval()
    real_B = real_B.to(gan.device)
    fake_B2A, _, fake_B2A_heatmap = gan.genB2A(real_B)
    # fake_B2A2B, _, fake_B2A2B_heatmap = gan.genA2B(fake_B2A)
    # fake_B2B, _, fake_B2B_heatmap = gan.genA2B(real_B)
    B2A = 255.0 * tensor2numpy(denorm(fake_B2A[0]))

    cv2.imwrite(os.path.join('result', 'B2A', 'B2A_{}.png'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime()))), RGB2BGR(B2A))
    return B2A


@app.route('/generate', methods=['POST'])
def generate():
    print('[APP] Predict')
    result = {'success': False}

    if flask.request.method == 'POST':
        data = flask.request.form
        print('data:', data)
        if flask.request.files.get('image'):
            image_buffer = flask.request.files['image'].read()

            task = data['task']
            print('task:', task)

            if task == 'A2B':
                image_A2B = A2B_infer(image_buffer)
                result['generate'] = image_A2B.tolist()
            elif task == 'B2A':
                image_B2A = B2A_infer(image_buffer)
                result['generate'] = image_B2A.tolist()

            result['success'] = True
            
            torch.cuda.empty_cache()

    # print('result:', result)
    return flask.jsonify(result)


"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=True, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='Landscape', help='dataset_name')

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

    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--test_iter', type=int, default=900000, help='Test load model iteration')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    # check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    # check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    # check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

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


class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.test_iter = args.test_iter

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# test_iter :", self.test_iter)
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        # self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        # self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        # self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        # self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        # self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        # self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        # self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def load(self, dir, step):
        model_path = os.path.join(dir, self.dataset + '_params_%07d.pt' % step)
        print('model_path:', model_path)
        params = torch.load(model_path)
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test2(self): 
        A2B_path = os.path.join(self.result_dir, self.dataset, 'test', str(self.test_iter), 'A2B')
        B2A_path = os.path.join(self.result_dir, self.dataset, 'test', str(self.test_iter), 'B2A')
        if not os.path.exists(A2B_path):
            os.makedirs(A2B_path)
        if not os.path.exists(B2A_path):
            os.makedirs(B2A_path)
        
        
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            # model_list.sort()
            # iter = int(model_list[-1].split('_')[-1].split('.')[0])
            iter = self.test_iter
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()

        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            # fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            # fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                #   cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                #   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                #   cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                #   cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                #   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))
                                ), 0)
            
            cv2.imwrite(os.path.join(A2B_path, 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            # fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            # fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                #   cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                #   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                #   cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                #   cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                #   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))
                                ), 0)

            cv2.imwrite(os.path.join(B2A_path, 'B2A_%d.png' % (n + 1)), B2A * 255.0)


    def load_model(self): 
        try:
            self.load(os.path.join('..', self.result_dir, self.dataset, 'model'), self.test_iter)
            print(" [*] Load SUCCESS")
        except:
            print(" [*] Load FAILURE")


if __name__ == '__main__':
    time_start = time.time()
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    gan.load_model()

    time_end = time.time()
    print('Server start time: {:.3f}s'.format(time_end-time_start))
    
    app.run(host='0.0.0.0', port=5005, debug=False)
