#!/bin/bash

# Total number of experiments
TOTAL=100

# Maximum number of concurrent processes
CONCURRENT=40

# Starting index (adjust as needed)
START_INDEX=1

# Calculate the ending index
END_INDEX=$((START_INDEX + TOTAL - 1))

# Loop over the range of experiment indices
for ((i=START_INDEX; i<=END_INDEX; i++))
do
    # Start the Python script in the background and redirect output
    python3 main.py "$i" > "out_$i" &

    # Get the current number of background jobs
    CURRENT_JOBS=$(jobs -rp | wc -l)

    # If the number of jobs reaches the concurrency limit, wait for any to finish
    if [ "$CURRENT_JOBS" -ge "$CONCURRENT" ]; then
        wait -n  # Wait for the next job to finish
    fi
done

# Wait for all remaining background jobs to complete
wait

echo "All $TOTAL experiments have completed."
