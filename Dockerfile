
ARG BASE_IMAGE
ARG ROLE

FROM $BASE_IMAGE

WORKDIR /fl_training

# Create a placeholder file based on ROLE
RUN touch placeholder_file

# Copy role-specific files based on ROLE
COPY server.py /fl_training/server.py
COPY main_server.py /fl_training/main_server.py

# Check if ROLE is "client" and remove unnecessary files
RUN if [ "$ROLE" = "client" ]; then \
        rm -f server.py main_server.py placeholder_file; \
    else \
        rm -f client.py placeholder_file; \
    fi

ADD config /fl_training/config
#ADD client.py /client/client.py
COPY utils.py /fl_training/utils.py
COPY models.py /fl_training/models.py
COPY requirements.txt /fl_training/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash"]