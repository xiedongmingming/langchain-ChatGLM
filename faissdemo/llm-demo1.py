from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.schema import HumanMessage

import os

os.environ['OPENAI_API_KEY'] = 'xxx'
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

resp = llm("Write me a song about sparkling water.")


llm.generate(["Tell me a joke."])