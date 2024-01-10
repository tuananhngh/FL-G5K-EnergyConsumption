# Studying the energy consumption of Federate Learning
The goal if this project is to study the energy consumption of Federated Learning.

## FL on Grid'5000
We use the French experimental platform called Grid'5000 which contains a variety of machines from A100 clusters to Jetson Xavier AGX.
Below are the steps required to execute the FL algorithm on Grid'5000 using the Flower framework.

### Book as many nodes as needed
```
flyon$ oarsub -l host=3 -p chifflot -I
```

### Install without yaml file
```
flyon$ module load conda
flyon$ conda create --name your_environment_name
flyon$ conda activate your_environment_name
flyon$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
(or flyon$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)
flyon$ pip3 install -q flwr[simulation] flwr_datasets[vision] tqdm matplotlib hydra-core
```
### Lanch server and clients on same node
```
flyon$ ssh node-server (for exemple chifflot-2)
node-server$ module load conda
node-server$ conda activate fl
node-server$ bash run_server.sh
```
### Launch server and clients on different nodes
```
flyon$ ssh node-server (for exemple chifflot-2)
node-server$ module load conda
node-server$ conda activate your_environment_name
node-server$ hostname -I (ou IP_SERVER=`hostname -I` puis echo $IP_SERVER)
```
This returns the server ID. 
Copy it into the host parameter in [config.yaml](./config/config_file.yaml).
```
node-server$ python main_server.py comm.host=$IP_SERVER
```
For every client to be launched:
```
flyon$ ssh node-client (for exemple chifflot-4)
node-client$ module load conda
node-client$ conda activate your_environment_name
node-client$ cd FL-G5K-Test
node-client$ python client.py client_params.client_id=1 comm.host=$IP_SERVER 
```
The server will start when enough client are connected (as defined in the config file).

### Using docker
Install docker on your node:
```
g5k-setup-nvidia-docker -t
```

C'est plus simple d'utiliser les images prédéfinies.

Client Image : docker pull slatonnguyen410/flower-client-jetson:1.0
Run docker : docker run --network host --rm -it slatonnguyen410/flower-client-jetson:1.0
Inside container : meme command pour client comme précédent.

Server Image : docker pull slatonnguyen410/flower-server-jetson:1.0
Run docker : docker run --network host --rm -it slatonnguyen410/flower-server-jetson:1.0
Inside container : meme command pour server comme précédent.
Il faut que client et server soient sur le meme réseau. 

```
bash run_docker.sh -b jetson
```