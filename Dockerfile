
ARG BASE_IMAGE

FROM $BASE_IMAGE

WORKDIR /fl_training

# Copy role-specific files based on ROLE
COPY server.py /fl_training/server.py
COPY main_server.py /fl_training/main_server.py

ADD config /fl_training/config
COPY utils.py /fl_training/utils.py
COPY models.py /fl_training/models.py
COPY requirements.txt /fl_training/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash"]