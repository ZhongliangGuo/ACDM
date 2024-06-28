import os
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from efdm import EFDM
from lpips import LPIPS
from pathlib import Path
from torch.utils import data
from metrics.SSIM import SSIM
from torchvision import transforms
from argparse import ArgumentParser
from metrics.ACDM import AestheticColorDistanceMetric


def init_seeds(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed(3407)
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, device):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img).to(self.device)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def get_data_iter(device, data_dir, data_transform=None, batch_size=16, num_workers=16):
    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ])

        dataset = FlatFolderDataset(data_dir, data_transform, device)
        return iter(data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=InfiniteSamplerWrapper(dataset),
                                    num_workers=num_workers))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--vgg_path', type=str, default='./vgg_normalised.pth')
    parser.add_argument('--efdm_decoder_path', type=str, default='./efdm_decoder.pth')
    parser.add_argument('--content_path', type=str, default='/.../datasets/COCO')
    parser.add_argument('--style_path', type=str, default='/.../datasets/PainterByNumber')
    parser.add_argument('--random_seed', type=int, default=3407)
    args = parser.parse_args()
    device = torch.device('cuda')
    nst = EFDM(device, args.vgg_path, args.efdm_decoder_path)
    acdm = AestheticColorDistanceMetric()
    ssim = SSIM()
    lpips_loss = LPIPS(net='vgg').cuda()
    init_seeds(args.random_seed)
    content_iter = get_data_iter(device='cuda',
                                 data_dir=args.content_path,
                                 batch_size=1,
                                 num_workers=0)
    style_iter = get_data_iter(device='cuda',
                               data_dir=args.style_path,
                               batch_size=1,
                               num_workers=0)
    same_style_diff_content = {'acdm': [],
                               'lpips': [],
                               'ssim': []}
    same_content_diff_style = {'acdm': [],
                               'lpips': [],
                               'ssim': []}
    pbar = tqdm(total=args.num_samples)
    for i in range(args.num_samples):
        style_batch1 = next(style_iter)
        style_batch2 = next(style_iter)
        content_batch1 = next(content_iter)
        content_batch2 = next(content_iter)
        c1s1 = nst(content_batch1, style_batch1)
        c2s1 = nst(content_batch2, style_batch1)
        c1s2 = nst(content_batch1, style_batch2)
        same_style_diff_content['acdm'].append(sum(acdm(c1s1, c2s1)))
        same_content_diff_style['acdm'].append(sum(acdm(c1s1, c1s2)))
        same_style_diff_content['lpips'].append(lpips_loss(c1s1, c2s1).item())
        same_content_diff_style['lpips'].append(lpips_loss(c1s1, c1s2).item())
        same_style_diff_content['ssim'].append(ssim(c1s1, c2s1).item())
        same_content_diff_style['ssim'].append(ssim(c1s1, c1s2).item())
        pbar.update(1)
    pbar.close()
    for metric_name in same_style_diff_content.keys():
        positive_score = same_style_diff_content[metric_name]
        negative_score = same_content_diff_style[metric_name]
        print(f'{metric_name}:', ' positive pair ', sum(positive_score) / len(positive_score),
              ' negative pair', sum(negative_score) / len(negative_score))


if __name__ == '__main__':
    main()
