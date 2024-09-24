#!/bin/bash

# Define the start and end index
START_INDEX=1
END_INDEX=30

# Loop over the range
for i in $(seq $START_INDEX $END_INDEX)
do
    # Call the Python script with the current index
    python3 main.py $i
done


# commonpar,
# width=1.2\linewidth,
# height=.9\linewidth,