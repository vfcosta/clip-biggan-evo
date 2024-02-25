CMA-ES only:
```
python cma-es-engine.py --local-search-steps 0 --pop-size 10 --image-size 128 --lamarck
```

Adam only:
```
python cma-es-engine.py --local-search-steps 5 --pop-size 1 --image-size 128
```

Hybrid:
```
python cma-es-engine.py --local-search-steps 5 --pop-size 10 --image-size 128 --lamarck
```


GEN-TSNE

Extract features:
```
python -m gen_tsne.features_extractor -p "../clip-biggan-evo/2024/experiments/*/29_best.png" -m clip
```

Run TSNE (or another method)
```
python main.py -p "../clip-biggan-evo/2024/experiments/*/29_best.png" -f -d tsne --rows 30 --cols 30 -j 2
```


Run CMA with map fitness
```
CUDA_VISIBLE_DEVICES= python cma-es-engine.py --local-search-steps 1 --pop-size 5 --image-size 128 --max-gens 30 --use-map-fitness --reference-image 2024/experiments/a_painting_of_superman_by_van_gogh_clip_cond_vector_64_30_10_0.2_5_v52/29_best.png
```
