FROM ubuntu:20.04
ENV USER_PATH=/home/appuser

RUN useradd --create-home appuser
RUN apt-get update && apt-get -y install sudo rustc
RUN apt install -y python3-pip build-essential libssl-dev libffi-dev python3-dev
USER appuser
EXPOSE 5000
WORKDIR "$USER_PATH/app"
COPY . "$USER_PATH/app"
ENV PATH "$PATH:$USER_PATH/.local/bin"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt