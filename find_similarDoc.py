from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from http import HTTPStatus
import dashscope

dashscope.api_key = "706536f8d325979c4d4fd5de104e3864.udrW9pscb7bmddLM"

chromadbPath_bge = "/Users/shaobin/Personal/In Hong Kong/Semester-B/Project/data/Medical_Insurance_Data_Collection/database_bge"
chromadbPath_stella = "../database_stella"
chromadbPath_peg = "../database_peg"
trdFilePath = "section_break_json"
model_name1 = "BAAI/bge-base-zh-v1.5"
model_name2 = "infgrad/stella-base-zh-v2"
model_name3 = "TownsWu/PEG"

# encode_kwargs = {"normalize_embeddings": True}
encode_kwargs = {"normalize_embeddings": True}
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name1,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

client = chromadb.PersistentClient(path=chromadbPath_bge)
tableName = "my_collection"
try:
    collection = client.get_collection(name=tableName)
    print("get collection")
except:
    collection = client.create_collection(name=tableName)
    print("create collection")
    
# query
print("get the query text and embedding...")
query_text = "保單持有人或受保人對任何註冊醫生、醫院或其他醫療服務提供者，因任何原因或理由所提出的損害進行訴訟或另類排解糾紛程序，公司是否應承擔責任？"
query_embedding = embeddings.embed_query(query_text)
print("length of queryEmbeddings are : " + str(len(query_embedding)))


print("querying...")
result = collection.query(
    query_embeddings=query_embedding,
    n_results=5
    # where={"metadata_field": "is_equal_to_this"},
    # where_document={"$contains":"search_string"}
)

print(result)
for i in range(len(result["ids"][0])):
    print(len(result['documents'][0][i]))


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [[query_text, result['documents'][0][0]], [query_text, result['documents'][0][1]], [query_text, result['documents'][0][2]], [query_text, result['documents'][0][3]], [query_text, result['documents'][0][4]]]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
    # reranker - NDCG
    # Fine the position of the highest score
    pos = torch.argmax(scores)






import time
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="706536f8d325979c4d4fd5de104e3864.udrW9pscb7bmddLM") # 请填写您自己的APIKey

def call_with_prompt_ZHIPU(prompt_in, client):    
    response = client.chat.asyncCompletions.create(
        model="glm-4-0520",  # 填写需要调用的模型名称
        messages=[
            {
                "role": "user",
                "content": prompt_in
            }
        ],
    )
    task_id = response.id
    task_status = ''
    get_cnt = 0
    while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 40:
        result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
        print(result_response)
        task_status = result_response.task_status

        time.sleep(2)
        get_cnt += 1





def call_with_prompt(prompt_in):
    rsp = dashscope.Generation.call(model='glm-4-0520',
                                    prompt=prompt_in,
                                    history=[])
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp.output)
        print(rsp.usage)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))


prefix = "请参考下列的知识并使用香港繁体中文回答问题"
knowledge = "[知识：]/n" + result['documents'][0][pos]
queryFinal = "[问题：]/n" + query_text
promptIn = prefix + knowledge + queryFinal

# call_with_prompt(promptIn)
call_with_prompt_ZHIPU(promptIn, client)