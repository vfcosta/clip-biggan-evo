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
python -m gen_tsne.features_extractor -p "../clip-biggan-evo/experiments/*/29_best.png" -m clip 
```

Run TSNE (or another method)
```
python main.py -p "../clip-biggan-evo/experiments/*/29_best.png" -f -d parametric_umap --rows 30 --cols 30
```
