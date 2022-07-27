In order to perform clusterization & selection (with all comercially available bromides stored in `all_aryls.csv`), use the command:

```
python cluster_balanced.py --radius 3 --mark -N 40 --out_core REPRO  all_aryls.csv
```

The file ```REPRO_clusters_r3_marked_selected.csv``` will contain the selection of bromides used in the study.
