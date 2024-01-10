
ARG BASE_IMAGE

FROM $BASE_IMAGE

WORKDIR /client

ADD config /client/config
ADD client.py /client/client.py
ADD utils.py /client/utils.py
ADD models.py /client/models.py
ADD requirements.txt /client/requirements.txt

RUN pip3 install --upgrade pip
# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -q flwr[simulation] flwr_datasets[vision] matplotlib
