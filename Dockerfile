
FROM nvcr.io/nvidia/pytorch:23.12-py3


WORKDIR /client

ADD config /client/config
ADD client.py /client/client.py
ADD utils.py /client/utils.py
ADD requirements.txt /client/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
