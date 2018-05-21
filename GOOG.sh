#!/bin/bash

set -x
set -e

# train(){
#     python3 cancel_order.py $1
#     python3 submission_order.py $1
#     python3 main.py ../GOOG_0817/ GOOG $1 E midspread > ../results_zheng/GOOG_$1_midspread.txt
# }

dates="080117A 080117B"
threshs="0 0.25 0.5 0.75 1"
for date in $dates
    do
    for thresh in $threshs
        do
            python3 main.py ../GOOG_0817/ GOOG $date A E $thresh 2 > ../results_zheng/GOOG_$date_$thresh.txt
        done
    done
wait
