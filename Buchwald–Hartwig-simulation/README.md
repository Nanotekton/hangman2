In orderr to perform the simulation (100 repetitions in parallel), run:

```python nextPointPredictor.py --predictor GPE  --conditions_selector PI --substrate_selector maxunc```  

It may be needed to kill the process if some of the subprocessses hangs.

Then, assuming the log file is named BH_GPE_PI_maxUnc.log, run the following command to extract the data:

```python log2csv_av_on_space.py --scatter --output_core GPE_PI_maxUnc_data --lines_total $(wc -l BH_GPE_PI_maxUnc.log|awk '{print $1}') BH_GPE_PI_maxUnc.log```

The resulting CSV file is ready for analyses.
