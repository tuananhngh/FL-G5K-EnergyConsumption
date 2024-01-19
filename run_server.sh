#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

server_ip="$1"

NVML_SENSOR_DIR="$(pwd)/nvml_sensor"
JETSON_SENSOR="$(pwd)/jetson_monitoring_energy.py"
mkdir "$(pwd)/monitoring_energy/"
RESULT_DIR="$(pwd)/monitoring_energy/$(date '+%Y/%m/%dT%H:%M:%S.%6N')"
PERIOD=0.5
sleep_before=30
sleep_after=30
# bash ${JETSON_SENSOR} ${RESULT_DIR} &
# jetson_pid=$!
# echo "Jetson sensor running with pid $jetson_pid"
echo ${NVML_SENSOR_DIR}
sudo ${NVML_SENSOR_DIR} --result-dir ${RESULT_DIR} --period-seconds ${PERIOD} &
sensor_pid=$!
echo "Intenal sensors running with pid $sensor_pid"

sleep $sleep_before
echo "start_server DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')"

python3 main_server.py comm.host=$server_ip

echo "end_Server DATE $(date '+%Y/%m/%dT%H:%M:%S.%6N')"
sleep $sleep_after

# kill $jetson_pid
sudo pkill nvml_sensor

rm -rf /data/ # Remove old data
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

