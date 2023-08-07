# Define the directory containing the configuration files
CONFIG_DIR="./configs/onestop/t5-small-boosted"
# Define the log file
LOG_FILE="./t5_small_boosted_onestop_pipeline_log.txt"

# Use the find command to search for configuration.json files
# Then read each file path with the while loop
find "$CONFIG_DIR" -name "configuration.json" | while read CONFIG_FILE; do
  # Log the start of processing this config file
  echo "Processing $CONFIG_FILE..." >> $LOG_FILE
  # Run the python script with the current configuration file
  python3 ./finetune_t5_trainer.py "$CONFIG_FILE"
  # Log the completion of processing this config file
  echo "Done processing $CONFIG_FILE." >> $LOG_FILE
done
