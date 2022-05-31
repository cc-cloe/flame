# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
执行yolo的yaml文件，看看网络对不对
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
主要看三个函数是在那里调用，谁调用谁！
def parse_model  class Model中调用，对yaml文件解析生搭建网络结构
class Model
class Detect

"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation  用于FLOPs计算
except ImportError:
    thop = None


# 对检测层的操作，在parse_model中使用了
# 用于连接detect层，将输入的feature map通过一整个卷积核公式计算到想要的shape，为nms做准备
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter  在export中这个参数设置为true

    # ch：三个输出feature map的Channel[128, 256, 512
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers 检测层个数：3
        self.na = len(anchors[0]) // 2  # number of anchors：每个feature map的anchor个数。每个anchor包括长和宽，所以//2
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 对输出的feature map调用
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
            train和inference返回的不同
            return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]

        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference->推理
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


# 模型搭建过程：前处理、推理、后处理
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml 解析yaml文件
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 读取yaml内容

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:  # 一般不执行
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:  # 不执行，一般传入anchors是none
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 创建网络模型 self.model初始化整个网络模型，包括detect.. self.save所有层结构中的from!=-1的序号排序
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names。[0,1]
        self.inplace = self.yaml.get('inplace', True)  # inplace=true不使用加速推理

        # Build strides, anchors
        m = self.model[-1]  # Detect()对detect层操作
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    # 前处理模块
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:  # 在test时候使用数据增强
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
            x 输入图像
            profile True做性能评估
            visualize可视化处理
            return
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer，包括cancat和detect
                # 不等于-1可能是来自某一个层，或者多个层，[[[]]]，x就是多数数组的叠加了
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run执行每一层的forward
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


# 调用model中传来的字典。准确的说也就是打印打印那一堆，解析参数之后生成[input_channel, output_channel, args]
def parse_model(d, ch):  # model_dict, input_channels(3) ch记录每一层的输出channel。初始[3]。然后是多个list[[],[],[]]用index搜索
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args  f是个数字某一层的index
        m = eval(m) if isinstance(m, str) else m  # eval strings。模块名称
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, CBAM, SELayer, CoordAtt, Res_CoordAtt, BottleneckSE, C3SE, RepVGGBlock]:
            c1, c2 = ch[f], args[0]  # c1输入channel，c2当前层输出channel  ch记录所有层的输出channel
            if c2 != no:  # if not output：不是输出层，不用控制宽度
                c2 = make_divisible(c2 * gw, 8)  # v5s  c2*0.5

            args = [c1, c2, *args[1:]]  # 输出的时候多了当前层的channel
            # 如果当前层是BottleneckCSP/C3/C3TR, 则需要在args中加入bottleneck的个数
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3SE]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # c2是输出的channel吧
        elif m is BiConcat:
            c2 = max([ch[x] for x in f])
            channel1, channel2 = args[0], args[1]
            channel1 = make_divisible(channel1 * gw, 8) if channel1 != no else channel1
            channel2 = make_divisible(channel2 * gw, 8) if channel2 != no else channel2
            args = [channel1, channel2]
        elif m is Detect:  # detect层的检测
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # elif m is SELayer:  # 加入SE模块
        #     channel, re = args[0], args[1]
        #     channel = make_divisible(channel * gw, 8) if channel != no else channel
        #     args = [channel, re]
        # elif m is SimAM:  # SimAM
        #     args = args
        elif m is ECA_Layer:  # 加入SE模块
            channel, k = args[0], args[1]
            channel = make_divisible(channel * gw, 8) if channel != no else channel
            args = [channel, k]
        elif m is SCse:  # 加入SE模块
            channel = args[0]
            channel = make_divisible(channel * gw, 8) if channel != no else channel
            args = [channel]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    """
    conda activate torch110
    cd /home/data/yolov5_6

    # 第一波
    python train.py --data aa_d_fire.yaml --weight "" --cfg yolov5s.yaml --img 640  --name d-fire --device 3
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-bi.yaml --img 640  --name d-fire --device 2
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-p2-anchor4.yaml --img 640  --name d-fire --device 1
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-p2-anchor3.yaml --img 640  --name d-fire --device 0
    
    # 第二波
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-p2-anchor4-bi.yaml --img 640  --name d-fire-p2-4bifpn --device 3
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-p2-anchor3-bi.yaml --img 640  --name d-fire-p2-3bifpn --device 2
    python train.py --data aa_d_fire.yaml --weight "" --cfg yolov5m.yaml --img 640  --name d-fire-v5m --device 1
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/buzhidao.yaml --img 640  --name d-fire-buzhidao --device 0
    d-firep2bifpn  # p2检测头+bifpn(2)
    
    # 第三波
    用超参数进化做假bifpn
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-bi.yaml --name d-fire-bifpn-evolve --device 3
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-3se.yaml --name d-fire-3se  --device 2
    
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-neck-attention-SCse-back.yaml --name d-fire-neck-attention-SCse-back  --device 3
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-neck-attention-SCse-front.yaml --name d-fire-neck-attention-SCse-front  --device 2
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-neck-attention-SCse-resnet.yaml --name d-fire-neck-attention-SCse-resnet  --device 1
    python train.py --data aa_d_fire.yaml --weight "" --cfg mymodel/yolov5-neck-attention-CA-resnet.yaml --name d-fire-neck-attention-CA-resnet  --device 0
    
    python train.py --data aa_d_fire.yaml --weight "" --hub myhub.yaml --cfg yolov5s.yaml --name focal-loss  --device 1
    
        1. yolov5s 
        2. mymodel/yolov5-bi.yaml
        3. mymodel/yolov5-p2-anchor4.yaml
        4. mymodel/yolov5-p2-anchor3.yaml
        5. mymodel/yolov5-p2-anchor4-bi.yaml
        6. mymodel/yolov5-p2-anchor3-bi.yaml
        7. yolov5m
        8. buzhidao
        9. mymodel/yolov5-bi.yaml  进化了超参数
        10. 加了3个SE
        11. v5s的img-size=512
        
        mymodel/yolov5-ConvTransposed.yaml

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='mymodel/yolov5-C3SE.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
