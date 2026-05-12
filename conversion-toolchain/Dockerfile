FROM python:3.11-slim

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip tzdata \
        libxcb1 libx11-6 libgl1 libglib2.0-0 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG DX_COM_VERSION=2.3.0
RUN pip install --no-cache-dir \
    "https://sdk.deepx.ai/release/dxcom/v${DX_COM_VERSION}/dx_com-${DX_COM_VERSION}-cp311-cp311-manylinux_2_31_x86_64.whl"

WORKDIR /app

COPY scripts/ /app/scripts/

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
