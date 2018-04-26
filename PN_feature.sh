#!/bin/bash

set -x
set -e

train(){
    python3 cancel_order.py $1
    python3 submission_order.py $1
    python3 main.py $1 E spread > ../results_zheng/$1.txt
}

dates="080116 080216 080316 080416 080516 080816 080816 080916 081016 081116 081216 081516 081616"
for date in $dates
do 
    train $date
done
wait
