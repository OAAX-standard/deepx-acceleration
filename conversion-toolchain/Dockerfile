FROM ubuntu:22.04

# Set TimeZone
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y passwd libgl1 libglib2.0-0 unzip tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY dx_com /app/dx_com
COPY scripts/ /app/scripts

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
