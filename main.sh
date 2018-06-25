#!/bin/bash
#PBS -l cput=1000:00:00
#PBS -l walltime=1000:00:00
#PBS -l mem=8gb

use anaconda3
use gcc48
KERAS_BACKEND=tensorflow python /user/i/iaraya/Wind_speed/main_gpu.py $1 $2 $3 $4
