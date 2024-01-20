#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

USER="tunguyen"
JETSON_SENSOR="$(pwd)/jetson_monitoring_energy.py"
#RESULT_DIR="/home/${USER}/FL-G5K-Test/monitoring_energy/$(date '+%Y_%m_%d/%H_%M_%S')/"
RESULT_DIR="$(pwd)/outputs/main_server_0/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')/"
mkdir -p $RESULT_DIR
#TMP_RESULT_DIR="/tmp/results_energy/$(date '+%Y_%m_%d/%H_%M_%S')/"
TMP_RESULT_DIR="/tmp/results_energy/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')/"
mkdir -p $TMP_RESULT_DIR
RESULT_ENERGY_CSV="energy.csv"
LOG_FILE="logs.log"
# PERIOD=0.5
sleep_before=3
sleep_after=3

function cleanup()
{
    echo "end_server DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')" 2>&1 | tee -a "${TMP_RESULT_DIR}${LOG_FILE}"
    sleep $sleep_after

    cp -r $TMP_RESULT_DIR/* $RESULT_DIR
    rm -rf $TMP_RESULT_DIR

    echo "Copied and rm tmp file to ${RESULT_DIR}"

    kill $jetson_pid
    sudo pkill nvml_sensor

    echo "Killed all background processes"
}

# START MONITORING
python3 ${JETSON_SENSOR} --log-dir ${TMP_RESULT_DIR} --log-csv ${RESULT_ENERGY_CSV} 2>&1 | tee -a "${TMP_RESULT_DIR}${LOG_FILE}" &
jetson_pid=$!
echo "Jetson sensor running with pid $jetson_pid"

# SLEEP
sleep $sleep_before
echo "start_server DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')" 2>&1 | tee -a "${TMP_RESULT_DIR}${LOG_FILE}"

# START SERVER AND ENABLE CLEANUP FOR CTRL-C 
trap cleanup SIGINT
python3 main_server.py hydra.run.dir=$RESULT_DIR
cleanup

# rm -rf /data/ # Remove old data
# # Enable CTRL+C to stop all background processes
# trap "trap - SIGTERM && echo "end_server DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')" && sleep $sleep_after && kill -- -$$" SIGINT SIGTERM
# wait

