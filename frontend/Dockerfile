# 使用 Node.js 20（CRA 推荐）
FROM node:24.13.0-slim

# 设置工作目录
WORKDIR /app

# 复制依赖清单（利用 Docker 层缓存）
COPY package*.json ./

# 安装依赖（可选：使用淘宝镜像加速）
RUN npm config set registry http://registry.npmmirror.com && \
    npm install

# 复制源代码
COPY . .

# 暴露 CRA 默认端口
EXPOSE 3000

# 关键：设置 HOST=0.0.0.0，让容器外可访问
ENV HOST=0.0.0.0

# 启动开发服务器
CMD ["npm", "start"]