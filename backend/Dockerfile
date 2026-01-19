FROM python:3.12-slim

WORKDIR /app
RUN mkdir -p tmp/
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
COPY ./backend/wheel/*   ./tmp/
RUN pip install --no-cache-dir  --find-links ./tmp/ torch
RUN rm -r ./tmp
RUN pip3 install --upgrade pip setuptools wheel

COPY ./backend/requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

# 复制代码和资源
RUN mkdir -p backend/
COPY ./backend/app.py ./backend/
COPY ./backend/translator.py ./backend/
RUN mkdir -p src/
COPY ./src/ ./src/

RUN ls -R /app

# 暴露端口（FastAPI 默认 8000）
EXPOSE 8000
ENV PATH="/usr/local/bin:${PATH}"
# 启动命令（使用 Uvicorn）
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]