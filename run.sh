#!/bin/bash

# Define the start and end index
# START_INDEX=100
# END_INDEX=120

START_INDEX=1
END_INDEX=40
# Loop over the range
for i in $(seq $START_INDEX $END_INDEX)
do
    # Call the Python script in a new shell
    bash -c "python3 main.py $i" > "out_$i" &
    
    # Sleep for 1 second before starting the next process
    sleep 1
done

# Wait for all background processes to finish
wait



# commonpar,
# width=1.2\linewidth,
# height=.9\linewidth,