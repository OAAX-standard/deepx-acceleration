FROM ubuntu:22.04

# Set TimeZone
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends passwd libgl1 libglib2.0-0 unzip tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/ /app/scripts
COPY dx_com/ /app/dx_com/

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
