## Submission
--------

1. Put your prediction files to this folder. Naming convention is `{language code}_{small/large}.{dev/test}`.

2. Run evaluate.py (official evaluation script)

```bash
python3 evaluate.py ../../FST/submission ../part1/development_languages/ --evaltype dev 
python3 evaluate.py ../../FST/submission ../part1/development_languages/ --evaltype test
```
