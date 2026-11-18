# Blind-test submission folder

At blind-test time, write these files here:

```text
ansh__A2.csv
ansh__B2.csv
ansh__C.csv
```

Example commands:

```powershell
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_A.yaml --output submission/ansh__A2.csv --no-labels --embeddings --smoothing
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_B.yaml --output submission/ansh__B2.csv
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_C.yaml --output submission/ansh__C.csv
```

Only the new C config and these three outputs should be pushed at blind test; no source code changes are required.
