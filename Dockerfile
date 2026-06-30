# 1. 选基础系统（和本地对齐，用 3.10）
FROM python:3.10-slim

# 2. 装 C++ 编译环境（针对 PyMuPDF 和 bge 模型）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 建工作文件夹
WORKDIR /app

# 4. 把需求清单拷进去
COPY requirements.txt ./

# 5. 用最原生的 pip 安装所有依赖
# 这里加了国内镜像源，保证下载速度飞起
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 把核心代码拷进去
COPY . .

# 7. 跑起来（你的主入口是 Streamlit，需要这样启动）
CMD ["streamlit", "run", "tests/test_agent.py"]