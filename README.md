# Federated Learning with Jetson Devices ion Grid5000

## Overview
This repository provides resources and scripts to set up a Federated Learning environment using Jetson devices on the Grid5000 cluster.

## Server Setup
1. **Reserve an Estats node for the server:**
    ```bash
    oarsub -I -t deploy -t exotic -q testing -p estats -l nodes=1,walltime=3
    ```

2. **Deploy the server:**
    ```bash
    kadeploy3 -a  ~/public/ubuntu-estats.dsc (change the path to your image)
    ```
3. **In another terminal, reserve nodes for clients:**
    ```bash
    oarsub -I -t deploy -t exotic -q testing -p estats -l nodes=$num_client,walltime=3
    ```
    Choose a appropriate value for $num_clients (reserved nodes)
4. **Deploy clients:**
    ```bash
    kadeploy3 -a  ~/public/ubuntu-estats.dsc
    ```

5. **After deployment, on the server side:**

     ```bash
     bash estats_deploy_server.sh $num_clients
     ```
    The experiment results will be saved in the folder ```./results``` of your frontend
7. **On the client side, run with the server's IP address:**
     ```bash
     bash estats_deploy_clients.sh $SERVER_IP $num_clients
     ```
>>>>>>> 56e21959063be58cb7d27053c9eda7d5644b0bcd
