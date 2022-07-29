## cureently running on hadoop server
conda activate ai-certification-dev-env
function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}
n_procs=20 #  max number of parallel processes

for model in full base; do
  for labormarket_bias in 0 2; do
  for delta_T_u in 5 10 20; do
    for class_boundary in 5 10 15; do
       python analyse_complex_model_batch.py --model $model --labormarket_bias $labormarket_bias --class_boundary $class_boundary --delta_T_u $delta_T_u &
     pwait $n_procs
    done
    done
  done
done
