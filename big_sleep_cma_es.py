# -*- coding: utf-8 -*-
"""The Big Sleep: BigGANxCLIP.ipynb
Original file is located at
    https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR
"""

import logging

import clip
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from biggan import model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Load the model
perceptor, preprocess = clip.load('ViT-B/32')

seed = 0
MAX_CLASSES = 0
CCOUNT = 0

"""# Parameters

Don't bother changing anything but the prompt below if you're not using a different type of BigGAN
"""

im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape


nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
text_features = None
frase = None
cuts = None

CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA", CUDA_AVAILABLE)
DEVICE = torch.device('cuda') if CUDA_AVAILABLE else torch.device('cpu')

if CUDA_AVAILABLE:
    model = model.cuda().eval()
else:
    model = model.eval()


def init(text, cutn=128):
    global text_features, frase, cuts
    frase = text
    tx = clip.tokenize(text)
    text_features = perceptor.encode_text(tx.to(DEVICE)).detach().clone()  # 1 x 512

    cuts = []
    for ch in range(cutn):
        size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        cuts.append((offsetx, offsety, size))


def displ(img, it=0, pre_scaled=True, individual=0):
    global frase
    global seed

    img = np.array(img)[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48 * 4, 32 * 4)
    imageio.imwrite(frase + "-seed=" + str(seed - 1) + "-it=" + str(it) + "-ind=" + str(individual) + '.png',
                    np.array(img))
    # print(frase + "-seed="+str(seed-1)+"-it="+str(0) + '.png')
    # return display.Image(str(3)+'.png')
    return 0


"""# Latent coordinate

Choose a place to start in BigGAN (it'll be a dog. Probably a hound lol)
"""


class CondVectorParameters(torch.nn.Module):
    def __init__(self, ind_numpy, batch_size=16):
        super(CondVectorParameters, self).__init__()
        # aqui FALTA COLOCAR COM CUDA        
        reshape_array = ind_numpy.reshape(batch_size, 256)
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


def copiado(lats, classes):
    lats = torch.nn.Parameter(torch.tensor(lats[0]).repeat(16, 1).float().to(DEVICE))
    # aux_c=classes[0]
    classes = torch.nn.Parameter(torch.tensor(classes[0]).repeat(16, 1).float().to(DEVICE))
    return lats, classes


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


eps = 0
lats = 0
optimizer = 0


def save_individual(latent_space, file_name, pre_scaled=True):
    # if it == 0: os.mkdir(path_to_folder)
    al = model(*latent_space(), 1).cpu().detach().numpy()
    for img in al:
        img = np.array(img)[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        if not pre_scaled:
            img = scale(img, 48 * 4, 32 * 4)
        imageio.imwrite(file_name, np.array(img))


def save_individual_cond_vector(cond_vector, file_name, pre_scaled=True):
    # if it == 0: os.mkdir(path_to_folder)
    al = model(cond_vector(), 1).cpu().detach().numpy()
    for img in al:
        img = np.array(img)[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        if not pre_scaled:
            img = scale(img, 48 * 4, 32 * 4)
        imageio.imwrite(file_name, np.array(img))


def checkin_with_cond_vectors(loss, cond_vector, individual=0, itt=0):
    best = torch.topk(loss[2], k=1, largest=False)[1]
    with torch.no_grad():
        al = model(cond_vector(), 1)[best:best + 1].cpu().numpy()
    for allls in al:
        displ(allls, it=itt, individual=individual)


def evaluate(cond_vector_params):
    cond_vector = cond_vector_params()  # 16 x 256
    # input()
    # z = cenas[0]
    # classes = cenas[1]
    # z.data[1 :, :] = z.data[0]
    # classes.data[1:, :] = classes.data[0]
    out = model(cond_vector, 1)  # 1 x 3 x 512 x 512

    p_s = []
    # cutn = 128
    # for ch in range(cutn):
    #     size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
    #     offsetx = torch.randint(0, sideX - size, ())
    #     offsety = torch.randint(0, sideX - size, ())
    for offsetx, offsety, size in cuts:
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(apper, (224, 224), mode='nearest')
        p_s.append(apper)
    # convert_tensor = torchvision.transforms.ToTensor()
    into = torch.cat(p_s, 0)

    # into = torch.nn.functional.interpolate(out, (224,224), mode='nearest')

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

    return [lat_l, cls_l, -100 * torch.cosine_similarity(text_features, iii, dim=-1).mean()]


def evaluate_with_local_search(cond_vector_params, local_search_steps=5):
    local_search_optimizer = torch.optim.Adam(cond_vector_params.parameters(), .07)
    loss1 = evaluate(cond_vector_params)
    for i in range(local_search_steps):
        loss = loss1[0] + loss1[1] + loss1[2]
        # print('Zerar Grads')
        local_search_optimizer.zero_grad()
        # print('Computar Grads')
        loss.backward()
        # print('Aplicar Adam')
        local_search_optimizer.step()
        loss1 = evaluate(cond_vector_params)

    return loss1


def scale(img, *args):
    logger.warning("not implemented")
    return scale
