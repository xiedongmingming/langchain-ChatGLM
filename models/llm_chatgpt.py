import os

from abc import ABC

from langchain.llms.base import LLM

from typing import Optional, List

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer, AnswerResult)

template = """Question: {question}

Answer: Let's think step by step."""

os.environ["OPENAI_API_KEY"] = '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'
os.environ["OPENAI_API_BASE"] = 'https://ai.adwetec.cn/v1'  # 代理地址


class ChatGPT(BaseAnswer, LLM, ABC):
    #
    max_token: int = 2000

    temperature: float = 0.01

    top_p = 0.9

    checkPoint: LoaderCheckPoint = None

    history = []

    history_len: int = 10

    sllm_chain: LLMChain = None
    cllm_chain: LLMChain = None

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        #
        super().__init__()

        self.checkPoint = checkPoint

        prompt_template = PromptTemplate(template=template, input_variables=["question"])

        llm1 = OpenAI(
            model_name="text-ada-001",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            streaming=True,
        )

        self.sllm_chain = LLMChain(prompt=prompt_template, llm=llm1)

        llm2 = OpenAI(
            model_name="text-ada-001",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_token,
            streaming=False,
        )

        self.cllm_chain = LLMChain(prompt=prompt_template, llm=llm2)

    @property
    def _llm_type(self) -> str:
        #
        return "ChatGPT"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        #
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        #
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        #
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        #
        response = self.cllm_chain.run(prompt)

        history += [[prompt, response]]

        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response}

        return answer_result

    def generatorAnswer(
            self,
            query: str,
            history: List[List[str]] = [],
            streaming: bool = False
    ):
        #
        if not history:

            prompt = query

        else:

            prompt = ""

            for i, (old_query, response) in enumerate(history):
                #
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)

            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        response = self.sllm_chain.run(prompt)

        history += [[prompt, response]]

        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response}

        yield answer_result


if __name__ == "__main__":
    #
    llm = ChatGPT()

    last_print_len = 0

    for resp, history in llm._call("你好"):
        #
        print(resp[last_print_len:], end="", flush=True)

        last_print_len = len(resp)

    for resp, history in llm.generatorAnswer("你好", streaming=False):
        #
        print(resp)

    pass
