In order to perform the simulation (100 repetitions in parallel), run:
The `prediction_full_space_2021-Jul-22-02.49.28.csv` contains final measurments (`Yexp` column) with missing parts filled with the predictions from the final model (`Ypred` column).

```
python do_simulation_corr.py --ground_truth prediction_full_space_2021-Jul-22-02.49.28.csv\
                        --batch_schedule 36 36 72 72 72 72 72 \
                        --app _batch_schedule_cond18_Jul08 \
                        --n_cond 18
```
```
python do_simulation_corr.py --ground_truth prediction_full_space_2021-Jul-22-02.49.28.csv \
                        --batch_schedule 36 36 72 72 72 72 72 \
                        --app _batch_schedule_cond18_Jul08_totally_random \
                        --n_cond 18 \
                        --totally_random 
```

It may happen then one of subprocesses hangs - after killing the program, one can extract the data directly from the log file:

```
python extract_from_log.py --output <output_name>  <logfile>
```

The resulting CSV files are ready for analyses.

