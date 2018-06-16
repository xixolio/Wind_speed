#!/bin/bash
#PBS -l cput=1000:00:00
#PBS -l walltime=1000:00:00
#PBS -l mem=3gb

use anaconda3
use gcc48
KERAS_BACKEND=theano python /user/i/iaraya/CIARP/Wind_speed/main.py $1 $2 $3 $4
