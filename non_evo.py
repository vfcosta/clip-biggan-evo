import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import extra_tools
import argparse
from datetime import datetime
import logging

import big_sleep_cma_es

# Problem size
N_GENS = 125
NUM_LATENTS = None  # na verdade é o número de vetores latentes (z => layers_biggan + 1)  https://github.com/lucidrains/big-sleep/issues/34
IMAGE_SIZE = 512
Z_DIM = 128
RANDOM_SEED = 64
LOCAL_SEARCH_STEPS = 0
TEXT = "a painting of superman by van gogh"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_individual_with_embeddings():
    latent = torch.nn.Parameter(torch.zeros(NUM_LATENTS, Z_DIM).normal_(std=1).float().cuda())
    params_other = torch.zeros(NUM_LATENTS, 1000).normal_(-3.9, .3).cuda()
    classes = torch.sigmoid(torch.nn.Parameter(params_other))
    embed = big_sleep_cma_es.model.embeddings(classes)
    cond_vector = torch.cat((latent, embed), dim=1)
    logger.info("cond_vector shape: %s", cond_vector.shape)
    ind = cond_vector.cpu().detach().numpy().flatten()
    logger.info("individual shape: %s", ind.shape)
    return ind


def main():
    parser = argparse.ArgumentParser(description="evolve to objective")
    global RANDOM_SEED, N_GENS, LOCAL_SEARCH_STEPS, TEXT, IMAGE_SIZE, NUM_LATENTS

    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))

    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is 'experiments'.")
    parser.add_argument('--max-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--local-search-steps', default=LOCAL_SEARCH_STEPS, type=int, help='Local search steps. Default is {}.'.format(LOCAL_SEARCH_STEPS))
    parser.add_argument('--text', default=TEXT, type=str, help='Text for image generation. Default is {}.'.format(TEXT))
    parser.add_argument('--image-size', default=IMAGE_SIZE, type=int, help='Image size. Default is {}.'.format(IMAGE_SIZE))
    args = parser.parse_args()
    save_folder = args.save_folder
    N_GENS = int(args.max_gens)
    RANDOM_SEED = args.random_seed
    LOCAL_SEARCH_STEPS = args.local_search_steps
    TEXT = args.text
    IMAGE_SIZE = args.image_size
    experiment_name = f"{TEXT.replace(' ', '_')}_clip_cond_vector_{RANDOM_SEED or datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    sub_folder = f"{experiment_name}_{N_GENS}_1_0_{LOCAL_SEARCH_STEPS}"
    np.random.seed(RANDOM_SEED)
    model = big_sleep_cma_es.init(TEXT, image_size=IMAGE_SIZE)
    NUM_LATENTS = len(model.config.layers) + 1
    save_folder, sub_folder = extra_tools.create_save_folder(save_folder, sub_folder)
    with open(os.path.join(save_folder, sub_folder, "params.json"), "w") as f:
        json.dump(vars(args), f)

    print("params", args, f"num_latents={NUM_LATENTS}")

    ind = generate_individual_with_embeddings()
    cond_vector = big_sleep_cma_es.CondVectorParameters(ind, num_latents=NUM_LATENTS)
    for gen in range(N_GENS):
        loss = big_sleep_cma_es.evaluate_with_local_search(cond_vector, LOCAL_SEARCH_STEPS)
        print(gen, loss)
        big_sleep_cma_es.save_individual_image(cond_vector, f"{save_folder}/{sub_folder}/{gen}_best.png")


if __name__ == "__main__":
    main()
