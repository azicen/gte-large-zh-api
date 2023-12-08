# gte-large-zh-api

一个用于给`one-api`提供文本嵌入模型`gte-large-zh`的简单项目

测试通过的客户端：
- FastGPT
- 小幻助理


## 使用

### 环境变量
`API_KEY` 设置为自定义的sk-key，用于接口权限校验


### 使用容器运行（推荐）

提供了使用cpu和cuda的两种镜像，如果需要使用cuda，请将下面的镜像标签修改为`1.0.0-cuda12.1-cudnn8`

#### Docker
```bash
docker run -e API_KEY=sk-key \
           -e TZ="Asia/Shanghai" \
           -p 8000:8000 \
           ghcr.io/azicen/gte-large-zh-api:latest
```

#### Docker compose
```yaml
version: '3.2'

services:
  gte-large-zh-api:
    image: "ghcr.io/azicen/gte-large-zh-api:latest"
    container_name: gte-large-zh-api
    environment:
      TZ: "Asia/Shanghai"
      API_KEY: sk-key
    ports:
      - "8000:8000"
    # 如果需要使用cuda，请取消下面的注释
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: ["gpu"]
```


### 使用Python直接运行

如果需要使用cuda请确保你已经安装了相应的驱动

```
git clone https://github.com/azicen/gte-large-zh-api.git

cd gte-large-zh-api

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python embeddings-api.py
```
