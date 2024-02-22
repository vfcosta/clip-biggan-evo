# -*- coding: utf-8 -*-
"""The Big Sleep: BigGANxCLIP.ipynb
Original file is located at
    https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR

Also available at https://github.com/lucidrains/big-sleep
"""

import logging
import pickle

import clip
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from umap.parametric_umap import load_ParametricUMAP

from biggan import BigGAN

logger = logging.getLogger(__name__)

seed = 0
MAX_CLASSES = 0

"""# Parameters

Don't bother changing anything but the prompt below if you're not using a different type of BigGAN
"""

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
text_features = None
frase = None
cuts = None
im_shape = None
num_cuts = None
model = None

CUDA_AVAILABLE = torch.cuda.is_available()
logger.info("CUDA: %s", CUDA_AVAILABLE)
DEVICE = torch.device('cuda') if CUDA_AVAILABLE else torch.device('cpu')

# Load the model
CLIP_MODEL = 'ViT-B/32'
perceptor, preprocess = clip.load(CLIP_MODEL, DEVICE)


USE_MAP_FITNESS = False
MAP_POINT = np.array([3.9486639499664307, 3.5445408821105957])
reduction_model = None


def init(text, cutn=128, image_size=512):
    global text_features, frase, cuts, model, im_shape, num_cuts
    num_cuts = cutn
    frase = text
    tx = clip.tokenize(text)
    text_features = perceptor.encode_text(tx.to(DEVICE)).detach().clone()  # 1 x 512

    model = BigGAN.from_pretrained(f'biggan-deep-{image_size}')
    model = model.cuda().eval() if CUDA_AVAILABLE else model.eval()

    im_shape = [image_size, image_size, 3]
    return model


class CondVectorParameters(torch.nn.Module):
    def __init__(self, ind_numpy, num_latents=15):
        super(CondVectorParameters, self).__init__()
        reshape_array = ind_numpy.reshape(num_latents, -1)
        self.normu = torch.nn.Parameter(torch.tensor(reshape_array).float().to(DEVICE))
        self.thrsh_lat = torch.tensor(1).to(DEVICE)
        self.thrsh_cls = torch.tensor(1.9).to(DEVICE)

    #  def forward(self):
    # return self.ff2(self.ff1(self.latent_code)), torch.softmax(1000*self.ff4(self.ff3(self.cls)), -1)
    #   return self.normu, torch.sigmoid(self.cls)

    # def forward(self):
    #     global CCOUNT
    #     if (CCOUNT < -10):
    #         self.normu,self.cls = copiado(self.normu, self.cls)
    #     if (MAX_CLASSES > 0):
    #         classes = differentiable_topk(self.cls, MAX_CLASSES)
    #         return self.normu, classes
    #     else:
    #         return self.normu#, torch.sigmoid(self.cls)
    def forward(self):
        return self.normu


def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x = x.scatter(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim=1)


def save_individual_image(cond_vector, file_name):
    al = model(cond_vector(), 1).cpu().detach().numpy()
    for img in al:
        img = np.array(img)[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(file_name, ((img + 1) * 127.5).astype(np.uint8))


def evaluate_map(images, use_features=True):
    global reduction_model
    if reduction_model is None:
        logger.info("loading dim_reduction model")
        with open("dim_reduction.pkl", "rb") as f:
            reduction_model = pickle.load(f)
        # reduction_model = load_ParametricUMAP("gen-tsne/model_parametric_umap")

    if use_features:  # use clip features
        features = perceptor.encode_image(
            torch.nn.functional.interpolate(images, (224, 224), mode='nearest').int()).detach().cpu().numpy()
    else:
        images = torch.nn.functional.interpolate(images, (128, 128), mode='nearest').int()
        features = images.detach().cpu().numpy().reshape(1, -1)
    points = reduction_model.transform(features)
    mean_distances = np.linalg.norm(points - MAP_POINT, axis=1).mean()  # calc euclidean distance between the two points
    return mean_distances


def evaluate(cond_vector_params):
    cond_vector = cond_vector_params()  # 16 x 256
    # input()
    # z = cenas[0]
    # classes = cenas[1]
    # z.data[1 :, :] = z.data[0]
    # classes.data[1:, :] = classes.data[0]
    out = model(cond_vector, 1)  # 1 x 3 x 512 x 512

    map_fitness = 0
    if USE_MAP_FITNESS:
        map_fitness = torch.tensor(evaluate_map(((out + 1) * 127.5)))

    p_s = []
    sideX, sideY, channels = im_shape
    for ch in range(num_cuts):
        size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        p_s.append(torch.nn.functional.interpolate(apper, (224, 224), mode='nearest'))
    # convert_tensor = torchvision.transforms.ToTensor()
    into = torch.cat(p_s, 0)

    into = nom(((into) + 1) / 2)
    iii = perceptor.encode_image(into)  # 128 x 512

    # llls = cond_vector_params()
    # lat_l = torch.abs(1 - torch.std(llls[0], dim=1)).mean() + torch.abs(torch.mean(llls[0])).mean() + 4 * torch.max(torch.square(llls[0]).mean(), lats.thrsh_lat)
    lat_l = 0

    # for array in llls[0]:
    #     mean = torch.mean(array)
    #     diffs = array - mean
    #     var = torch.mean(torch.pow(diffs, 2.0))
    #     std = torch.pow(var, 0.5)
    #     zscores = diffs / std
    #     skews = torch.mean(torch.pow(zscores, 3.0))
    #     kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    #     lat_l = lat_l + torch.abs(kurtoses) / llls[0].shape[0] + torch.abs(skews) / llls[0].shape[0]

    # cls_l = ((50*torch.topk(llls[1],largest=False,dim=1,k=999)[0])**2).mean()
    cls_l = 0

    cos_similarity = torch.cosine_similarity(text_features, iii, dim=-1).mean()
    return [lat_l, cls_l, -100 * cos_similarity, 100 * map_fitness]


def evaluate_with_local_search(cond_vector_params, local_search_steps=5, lr=.07):
    local_search_optimizer = torch.optim.Adam(cond_vector_params.parameters(), lr) if local_search_steps else None
    loss1 = evaluate(cond_vector_params)
    for i in range(local_search_steps):
        loss = loss1[0] + loss1[1] + loss1[2]
        local_search_optimizer.zero_grad()
        loss.backward()
        local_search_optimizer.step()
        loss1 = evaluate(cond_vector_params)
    return loss1
