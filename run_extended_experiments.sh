## cureently running on hadoop server

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}
n_procs=20 #  max number of parallel processes

for model in full base; do
  for labormarket_bias in 0 2; do
    #for delta_T_u in 10 20; do
    for class_boundary in 5 15; do
    #  python complex_model_run_scenarios_batch.py --model $model --labormarket_bias $labormarket_bias --delta_T_u $delta_T_u &
      python complex_model_run_scenarios_batch.py --model $model --labormarket_bias $labormarket_bias --class_boundary $class_boundary &
      pwait $n_procs
    done
  done
done
