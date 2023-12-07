import os
import torch
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi import FastAPI, HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from typing import Union
import logging

logger = logging.getLogger("uvicorn")

# 检测是否有GPU可用，如果有则使用cuda设备，否则使用cpu设备
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境变量
# cpu或cuda 用于在Dockerfile中传入 这将传递给SentenceTransformer
DEVICE = os.environ.get("DEVICE", device_type)
# token
API_KEY = os.environ.get("API_KEY", "sk-key")

if DEVICE == "cuda":
    logger.info(f"加载模型的设备为GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("加载模型的设备为CPU.")

# 可以修改成想要使用的其他模型
embeddings_model = SentenceTransformer("thenlper/gte-large-zh", device=DEVICE)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingProcessRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingQuestionRequest(BaseModel):
    input: str
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header:
        token_type, _, token = auth_header.partition(" ")
        if token_type.lower() == "bearer" and token == API_KEY:
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(
            expanded_embedding, (0, target_length - len(expanded_embedding))
        )
    return expanded_embedding


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings_v1(
    request: Union[EmbeddingProcessRequest, EmbeddingQuestionRequest]
):
    if isinstance(request, EmbeddingProcessRequest):
        logger.debug("EmbeddingProcessRequest")
        payload = request.input
    elif isinstance(request, EmbeddingQuestionRequest):
        logger.debug("EmbeddingQuestionRequest")
        payload = [request.input]
    else:
        logger.debug("Request")
        data = request.json()
        logger.debug(data)
        return

    logger.debug(payload)
    # 计算嵌入向量和tokens数量
    embeddings = [embeddings_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    # embeddings = [interpolate_vector(embedding, 1536) if len(embedding) < 1536 else embedding for embedding in embeddings]
    # 如果嵌入向量的维度不为1536，则使用特征扩展法扩展至1536维度
    # embeddings = [
    #     expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
    #     for embedding in embeddings
    # ]

    # Min-Max normalization
    # embeddings = [(embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)) if np.max(embedding) != np.min(embedding) else embedding for embedding in embeddings]
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
