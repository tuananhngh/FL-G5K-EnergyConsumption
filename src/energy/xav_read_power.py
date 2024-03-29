#!/usr/bin/env python
"""Use kwollect to get jetson data.

To use the results (example):
import requests

# Start the process
process = SshProcess("python3 /home/mjay/FL-G5K-Test/xav_read_power.py", host=server, connection_params={'user':'root'})
process.start()

# To get the data (later)
request_start = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() - 1000))
request_stop = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() -5))
node = "estats-5"

url = f"https://api.grid5000.fr/stable/sites/toulouse/metrics?nodes={node}&start_time={request_start}&end_time={request_stop}"
# url = "https://api.grid5000.fr/stable/sites/toulouse/metrics/?"
r = requests.get(url, verify=False).json()
"""
# https://docs.nvidia.com/jetson/archives/r35.2.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html#software-based-power-consumption-modeling

import glob
import os.path
import time
import requests
import datetime

SLEEP_TIME = 1.0

def format_val(timestamp, label, value):
    m = {
        "metric_id": f"jetson_{label}_power_watt",
        "timestamp": str(datetime.datetime.fromtimestamp(timestamp)),
        "value": value,
    }
    print(m)
    return m


entries = {}
for f in glob.glob("/sys/class/hwmon/*/*_label", recursive=True):
    label = open(f, "r").readline().strip().lower()
    if label == "sum of shunt voltages":
        continue
    dir = os.path.dirname(f)
    num = os.path.basename(f)[2]
    if not os.path.isfile(f"{dir}/in{num}_input") or not os.path.isfile(
        f"{dir}/curr{num}_input"
    ):
        continue
    entries[label] = {"dir": dir, "num": num}
print("HWMON files found:")
for label, v in entries.items():
    print(f"{label}: {v['dir']}/{{in,curr}}{v['num']}_input")

while True:
    time_start = time.time()
    tot = 0
    metrics = []
    for label, v in entries.items():
        volt = float(open(v["dir"] + f"/in{v['num']}_input", "r").readline().strip()) / 1000
        amp = float(open(v["dir"] + f"/curr{v['num']}_input", "r").readline().strip()) / 1000
        timestamp = time.time()
        power = volt * amp
        tot += power
        metrics.append(format_val(timestamp, label, power))
    timestamp = time.time()
    metrics.append(format_val(timestamp, "total", tot))
    r = requests.post("https://api.grid5000.fr/stable/sites/toulouse/metrics", json=metrics)
    time.sleep(SLEEP_TIME - (time.time() - time_start))
