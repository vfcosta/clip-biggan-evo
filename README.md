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
