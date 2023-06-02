import time
import logging
import requests
import json

from typing import Optional, List, Dict, Mapping, Any

import langchain

from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
from langchain.llms.utils import enforce_stop_tokens

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)

logging.basicConfig(level=logging.INFO)

langchain.llm_cache = InMemoryCache()  # 启动LLM的缓存


#
# 参考文档：https://zhuanlan.zhihu.com/p/624240080
#
class Callback(CallbackManagerForLLMRun):
    pass


class ChatGLM2(LLM):
    #
    url = "http://127.0.0.1:8000/chat"  # 模型服务URL

    stream = False

    history = []

    @property
    def _llm_type(self) -> str:
        #
        return "chatglm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None
    ) -> str:
        #
        response = requests.post(
            self.url,
            headers={
                'Content-Type': 'application/json'
            },
            data=prompt  # 这里的PROMPT已经是合成好的完整JSON串
        )

        if response.status_code != 200:
            return "查询结果错误"
        else:
            return response.json()

    # def _call(
    #         self,
    #         prompt: str,
    #         stop: Optional[List[str]] = None,
    #         run_manager: Optional[CallbackManagerForLLMRun] = None
    # ) -> str:
    #     #
    #     if self.stream:
    #
    #         headers = {"Content_Type": "application/json"}
    #
    #         data = {
    #             "prompt": "你好",
    #             "history": [],
    #             "max_length": 1024,
    #             "top_p": 3,
    #             "temperature": 0.2,
    #             "stream": True
    #         }
    #
    #         response = requests.post(self.url, headers=headers, json=data, stream=True)
    #
    #         for chunk in response.iter_content(chunk_size=1024):
    #             #
    #             # 处理响应内容
    #             #
    #             # chunk = json.loads(chunk)
    #             #
    #             # print(chunk)
    #
    #             yield chunk
    #
    #     else:
    #
    #         headers = {"Content_Type": "application/json"}
    #
    #         data = {
    #             "prompt": "你好",
    #             "history": [],
    #             "max_length": 1024,
    #             "top_p": 3,
    #             "temperature": 0.2,
    #             "stream": False
    #         }
    #
    #         response = requests.post(self.url, headers=headers, json=data, stream=False)
    #
    #         yield response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }

        return _param_dict


if __name__ == "__main__":

    llm = ChatGLM2()

    while True:
        #
        human_input = input("Human: ")

        begin_time = time.time() * 1000

        # 请求模型
        promot = {
            "prompt": human_input,
            "history": [],
            "max_length": 1024,
            "top_p": 3,
            "temperature": 0.2,
            "stream": False
        }

        response = llm(json.dumps(promot))

        end_time = time.time() * 1000

        used_time = round(end_time - begin_time, 3)

        logging.info(f"chatGLM process time: {used_time}ms")

        print(f"ChatGLM: {response}")
    #####################################################################
    #
    # import requests
    # import json
    #
    # url = 'http://127.0.0.1:8000/chat'
    #
    # headers = {"Content_Type": "application/json"}
    #
    # data = {
    #     "prompt": "你好",
    #     "history": [],
    #     "max_length": 1024,
    #     "top_p": 3,
    #     "temperature": 0.2,
    #     "stream": True
    # }
    #
    # # data = json.dumps(body)
    #
    # response = requests.post(url, headers=headers, json=data, stream=True)
    #
    # for chunk in response.iter_content(chunk_size=1024):
    #     #
    #     # 处理响应内容
    #     #
    #     chunk = json.loads(chunk)
    #
    #     print(chunk)
