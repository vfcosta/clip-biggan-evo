import torch
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from deap import base, cma, creator, tools
import extra_tools
import argparse
from datetime import datetime
import logging

import big_sleep_cma_es

# Problem size
N_GENS = 125
POP_SIZE = 10
NUM_LATENTS = None  # na verdade é o número de vetores latentes (z => layers_biggan + 1)  https://github.com/lucidrains/big-sleep/issues/34
IMAGE_SIZE = 512
Z_DIM = 128
COUNT_GENERATION = 0
RANDOM_SEED = 64
SAVE_ALL = False
LAMARCK = False
LOCAL_SEARCH_STEPS = 0
SIGMA = 0.2
TEXT = "a painting of superman by van gogh"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clip_fitness(individual):
    ind_array = np.array(individual)
    conditional_vector = big_sleep_cma_es.CondVectorParameters(ind_array, num_latents=NUM_LATENTS)
    result = big_sleep_cma_es.evaluate_with_local_search(conditional_vector, LOCAL_SEARCH_STEPS)
    if LAMARCK:
        individual[:] = conditional_vector().cpu().detach().numpy().flatten()
    return float(result[2].float().cpu()) * -1,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", clip_fitness)


def generate_individual_with_embeddings(num_latents, z_dim):
    latent = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std=1).float().cuda())
    params_other = torch.zeros(num_latents, 1000).normal_(-3.9, .3).cuda()
    classes = torch.sigmoid(torch.nn.Parameter(params_other))
    embed = big_sleep_cma_es.model.embeddings(classes)
    cond_vector = torch.cat((latent, embed), dim=1)
    logger.info("cond_vector shape: %s", cond_vector.shape)
    ind = cond_vector.cpu().detach().numpy().flatten()
    logger.info("individual shape: %s", ind.shape)
    return ind


def main(verbose=True):
    # The cma module uses the np random number generator
    parser = argparse.ArgumentParser(description="evolve to objective")
    global COUNT_GENERATION, RANDOM_SEED, N_GENS, POP_SIZE, SAVE_ALL, LAMARCK, LOCAL_SEARCH_STEPS, SIGMA, \
        TEXT, IMAGE_SIZE, NUM_LATENTS

    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))

    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is 'experiments'.")
    parser.add_argument('--max-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--local-search-steps', default=LOCAL_SEARCH_STEPS, type=int, help='Local search steps. Default is {}.'.format(LOCAL_SEARCH_STEPS))
    parser.add_argument('--sigma', default=SIGMA, type=float, help='Sigma for cma-es. Default is {}.'.format(SIGMA))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
    parser.add_argument('--lamarck', default=LAMARCK, action='store_true', help='Lamarckian evolution'.format(SAVE_ALL))
    parser.add_argument('--text', default=TEXT, type=str, help='Text for image generation. Default is {}.'.format(TEXT))
    parser.add_argument('--image-size', default=IMAGE_SIZE, type=int, help='Image size. Default is {}.'.format(IMAGE_SIZE))
    args = parser.parse_args()
    save_folder = args.save_folder
    POP_SIZE = int(args.pop_size)
    N_GENS = int(args.max_gens)
    RANDOM_SEED = args.random_seed
    SAVE_ALL = args.save_all
    LAMARCK = args.lamarck
    LOCAL_SEARCH_STEPS = args.local_search_steps
    SIGMA = args.sigma
    TEXT = args.text
    IMAGE_SIZE = args.image_size
    experiment_name = f"{TEXT.replace(' ', '_')}_clip_cond_vector_{RANDOM_SEED or datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    sub_folder = f"{experiment_name}_{N_GENS}_{POP_SIZE}_{SIGMA}_{LOCAL_SEARCH_STEPS}"
    np.random.seed(RANDOM_SEED)
    model = big_sleep_cma_es.init(TEXT, image_size=IMAGE_SIZE)
    NUM_LATENTS = len(model.config.layers) + 1
    save_folder, sub_folder = extra_tools.create_save_folder(save_folder, sub_folder)
    with open(os.path.join(save_folder, sub_folder, "params.json"), "w") as f:
        json.dump(vars(args), f)

    print("params", args, f"num_latents={NUM_LATENTS}")

    individual = generate_individual_with_embeddings(NUM_LATENTS, Z_DIM)

    if POP_SIZE == 1:
        logger.info("run non-evolutionary version")
        cond_vector = big_sleep_cma_es.CondVectorParameters(individual, num_latents=NUM_LATENTS)
        for gen in range(N_GENS):
            loss = big_sleep_cma_es.evaluate_with_local_search(cond_vector, LOCAL_SEARCH_STEPS)
            print(gen, loss)
            big_sleep_cma_es.save_individual_image(cond_vector, f"{save_folder}/{sub_folder}/{gen}_best.png")
        return

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    # strategy = cma.Strategy(centroid=np.random.normal(0.5, .5, GENOTYPE_SIZE), sigma=0.5, lambda_=POP_SIZE)
    strategy = cma.Strategy(centroid=individual, sigma=SIGMA, lambda_=POP_SIZE)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    for gen in range(N_GENS):
        COUNT_GENERATION = gen
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        COUNT_GENERATION = gen
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if SAVE_ALL or gen == N_GENS - 1:
            gen_folder = os.path.join(save_folder, sub_folder, str(gen))
            os.makedirs(gen_folder, exist_ok=True)
            for index, ind in enumerate(population):
                cond_vector = big_sleep_cma_es.CondVectorParameters(np.array(ind), num_latents=NUM_LATENTS)
                big_sleep_cma_es.save_individual_image(cond_vector, f"{gen_folder}/{index}.png")

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)

        if verbose:
            print(logbook.stream)

        if halloffame is not None:
            extra_tools.save_gen_best(save_folder, sub_folder, "experiment", [gen, halloffame[0], halloffame[0].fitness.values, "_"])
            cond_vector = big_sleep_cma_es.CondVectorParameters(np.array(halloffame[0]), num_latents=NUM_LATENTS)
            big_sleep_cma_es.save_individual_image(cond_vector, f"{save_folder}/{sub_folder}/{gen}_best.png")


if __name__ == "__main__":
    main(True)
