#!/bin/bash

# Output file to store the timing information
output_file="script_timings.txt"
echo "Script Execution Timings" > "$output_file"
echo "=========================" >> "$output_file"

# Function to run a script and record its timing
run_script() {
    local script="$1"
    echo "Running script: $script" | tee -a "$output_file"
    
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Start Time: $start_time" | tee -a "$output_file"
    
    # Run the script
    bash "$script"
    
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "End Time: $end_time" | tee -a "$output_file"
    
    # Calculate the duration
    start_sec=$(date -d "$start_time" +%s)
    end_sec=$(date -d "$end_time" +%s)
    duration=$((end_sec - start_sec))
    
    echo "Duration: $duration seconds" | tee -a "$output_file"
    echo "=========================" >> "$output_file"
}

# Running the scripts
# Long Term
# run_script "./scripts/long_term_forecast/Solar_script/TimeMixer_unify.sh"
# run_script "./scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh"
# run_script "./scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh"
# run_script "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh"
run_script "./scripts/long_term_forecast/Traffic_script/TimeMixer_unify.sh"
run_script "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh"
run_script "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh"
run_script "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh"
# Short Term
# run_script "./scripts/short_term_forecast/M4/TimeMixer.sh"
# run_script "./scripts/short_term_forecast/PEMS/TimeMixer.sh"

# Total duration
total_duration=0
while read line; do
    if [[ "$line" =~ "Duration" ]]; then
        duration=$(echo "$line" | awk '{print $2}')
        total_duration=$((total_duration + duration))
    fi
done < "$output_file"

echo "Total Duration: $total_duration seconds" >> "$output_file"
echo "=========================" >> "$output_file"

echo "All scripts completed. Check the timing details in $output_file."
