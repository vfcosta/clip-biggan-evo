from PIL import Image
import numpy as np
import pickle
from umap.parametric_umap import load_ParametricUMAP

import torch
import torchvision

from big_sleep_cma_es import calculate_map_points, DEVICE, init, CondVectorParameters

use_features = True
reference_image = "2024/experiments/a_painting_of_superman_by_van_gogh_clip_cond_vector_64_30_10_0.2_5_v52/29_best.png"


with open("../gen-tsne/dim_reduction.pkl", "rb") as f:
    reduction_model = pickle.load(f)
# reduction_model = load_ParametricUMAP("../gen-tsne/model_parametric_umap")
reference_image = torchvision.io.read_image(reference_image)
print(reference_image.shape)
MAP_POINT = calculate_map_points(reference_image.unsqueeze(0).to(DEVICE), reduction_model, use_features)[0]
print("MAP POINT", MAP_POINT)
inverse = reduction_model.inverse_transform([MAP_POINT])
# print("INVERSE", inverse, inverse.shape)
print("INVERSE POINT", reduction_model.transform(inverse))

if use_features:
    model = init("a painting of superman by van gogh", cutn=128, image_size=128)
    NUM_LATENTS = len(model.config.layers) + 1
    print("NUM_LATENTS", NUM_LATENTS)
else:
    image = inverse.reshape(128, 128, 3)
    print("Image", image.shape, image.min(), image.max())
    img = Image.fromarray(image, 'RGB')
    img.save("inverse_test.png")
