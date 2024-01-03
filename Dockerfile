
FROM python:3.8.5

#RUN mkdir -p /fl_app

WORKDIR /fl_app

COPY . /fl_app

RUN python -m pip install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt


