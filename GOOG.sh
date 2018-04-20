#!/bin/bash

set -x
set -e

train(){
    python3 cancel_order.py $1
    python3 submission_order.py $1
    python3 main.py ../GOOG_0817/ GOOG $1 E midspread > ../results_zheng/$1.txt
}

dates="080117 080217 080317 080417"
for date in $dates
do 
    train $date &
done
wait
