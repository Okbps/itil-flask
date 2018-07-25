FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential \
   && cd /usr/local/bin \
   && ln -s /usr/bin/python3 python \
   && pip3 install --upgrade pip
COPY . /app
WORKDIR /app
RUN pip3 install -r /app/requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]