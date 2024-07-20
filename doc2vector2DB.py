from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


chromadbPath_bge = "/Users/shaobin/Personal/In Hong Kong/Semester-B/Project/data/Medical_Insurance_Data_Collection/database_bge"
chromadbPath_stella = "/Users/shaobin/Personal/In Hong Kong/Semester-B/Project/data/Medical_Insurance_Data_Collection/database_stella"
chromadbPath_peg = "/Users/shaobin/Personal/In Hong Kong/Semester-B/Project/data/Medical_Insurance_Data_Collection/database_peg"
trdFilePath = "/Users/shaobin/Personal/In Hong Kong/Semester-B/Project/data/Medical_Insurance_Data_Collection/Notebooks/section_break_json"
model_name1 = "BAAI/bge-base-zh-v1.5"
model_name2 = "infgrad/stella-base-zh-v2"
model_name3 = "TownsWu/PEG"

# test true
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
except:
    collection = client.create_collection(name=tableName)
# collection = client.get_collection(name="my_collection")

# db = Chroma.from_documents(documents, embedding=embeddings)


with open(trdFilePath, "r", encoding="utf-8") as file:
    df_json = json.load(file)
    # print(df_json[0]['第八部分 釋義'])
    
    idList = []
    metaList = []
    docList = []

    # df_json中的每一个文件长度太大了，我要把它们分成512字数以内的小段，并且相同的文章拥有相同的id和meta
    print("Data starts to be processed...")
    key_i = 0
    for dic in df_json:
        id = str(dic['product_id'])
        
        doc = dic['第二部分 一般條件']
        # 如果加上最新的一句字数超过512，就把之前的句子作为一段文字加入doclist
        docs = doc.split("。")
        docTemp = ''
        i = 0
        for d in docs:
            if len(docTemp) + len(d) < 512:
                docTemp += d + "。"
            else:
                idList.append(str(key_i))
                metaList.append({"chunkID": i, "source": id})
                docList.append(docTemp)
                docTemp = d + "。"
                i = i + 1
                key_i = key_i + 1

    print("Data starts to be stored...")
    collection.add(
        documents=docList,
        embeddings=embeddings.embed_documents(docList),
        metadatas=metaList,
        ids=idList
    )

    collection.query(
        where={"ids": "1"},
    )