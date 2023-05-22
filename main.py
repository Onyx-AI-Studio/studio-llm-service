import gc
import os
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import pipeline
import boto3
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline

# Creating a Flask app
app = Flask(__name__)

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"
model_in_memory = ""
index_persist_directory = '/Users/snehalyelmati/Documents/studio-llm-service/indices'


# To check the if the API is up
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    if request.method == 'GET':
        data = "API is up!"
        return jsonify({'status': data})


@app.route('/llm_initialize', methods=['GET'])
def llm_initialize():
    if request.method == 'GET':
        global device, tokenizer, model, model_in_memory

        path = "google/flan-t5-base"
        model_in_memory = path
        # path = "bigscience/bloomz-1b7"
        # path = "bigscience/bloomz-3b"
        # path = "bigscience/bloomz-7b1"

        # path = "google/flan-t5-small"
        # path = "google/flan-t5-base"

        # path = "declare-lab/flan-alpaca-large"
        # path = "declare-lab/flan-gpt4all-xl"

        # path = "lmsys/vicuna-13b-delta-v1.1"
        # path = "AlekseyKorshuk/vicuna-7b"
        # path = "TheBloke/stable-vicuna-13B-GPTQ"
        # path = "TheBloke/stable-vicuna-13B-HF"
        # path = "TheBloke/stable-vicuna-13B.ggml.q4"

        # load default model
        load_model(path)

        return jsonify({'result': 'success'})


def load_model(path):
    global model, tokenizer, device, model_in_memory
    device = "mps"

    model_in_memory = path
    print(f'Setting model_in_memory to: {model_in_memory}')

    print(f'Loading model started from path: {path}')
    if "bloom" in path:
        tokenizer = BloomTokenizerFast.from_pretrained(path)
        model = BloomForCausalLM.from_pretrained(path)
        # Set the below parameter for low RAM usage
        # model = BloomForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
    elif "vicuna" in path:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        # model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        # model = AutoModelForSeq2SeqLM.from_pretrained(path, low_cpu_mem_usage=True)
    model.to(device)
    gc.collect()
    print(f'Loading model finished...')


# TODO: implement llm_save method functionality to handle other model types
@app.route('/llm_save', methods=['POST'])
def llm_save():
    if request.method == 'POST':
        print(f'Saving model started...')
        save_path = "/Users/snehalyelmati/Documents/models/"
        # TODO: replace all request.form to request.data
        model_name = request.form["model_name"]
        tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        model = BloomForCausalLM.from_pretrained(model_name)

        print(f'Save path: {save_path + model_name}')
        model.save_pretrained(save_path + model_name)
        tokenizer.save_pretrained(save_path + model_name)
        print(f'Saving model finished...')

        return jsonify({'result': 'success'})


def llm(input: str, is_full_prompt="True"):
    print("LLM prediction started...")
    # generator = pipeline("text2text-generation", model, tokenizer, device=device)
    # print(f'Pipeline output: {generator(input, max_length=150, num_return_sequences=1)}')

    if is_full_prompt == "True":
        inputs = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
    else:
        advice = input
        # advice = "Invest in company XYZ, the returns would surely triple in a year."
        inputs = tokenizer(
            f'''Assume you are a financial adviser who flags fraudulent advices. Your
task is to review the advice, delimited by <>, given by another
financial advisor to their client.

Question: Is the advice given by the financial adviser fraudulent?

Format your output as a valid JSON object with the following keys,

1. "Reasoning" - reasoning for the question above.
2. "Final answer" - final answer whether the advice is fraudulent. “Yes” if the advice is fraudulent, “No” if it is not fraudulent.

Do not include any additional information apart from the information that is requested above.

Advice: {advice}> 
Output:
'''
            , return_tensors="pt")["input_ids"].to(device)

    # inputs = tokenizer('A cat in French is "', return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(inputs, max_length=1000)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    print(f'LLM prediction finished!')
    return decoded_output


@app.route('/llm_predict', methods=['POST'])
def llm_predict():
    if request.method == 'POST':
        utterance = request.json["utterance"]
        llm_selected = request.json["llm_selected"]

        if llm_selected != model_in_memory:
            load_model(llm_selected)

        # is_full_prompt = request.form["is_full_prompt"]
        result = llm(utterance)
        return jsonify({'result': result})


@app.route('/build_indices', methods=['POST'])
def build_indices():
    if request.method == 'POST':
        s3_file_path = request.json["s3_file_path"]
        conv_id = request.json["conversation_id"]

        save_folder = '/Users/snehalyelmati/Documents/studio-llm-service/pdf_files'
        save_path = Path(save_folder, conv_id)
        print(f'save_path: {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = '_'.join(s3_file_path.split('/')[-1].split('_')[1:])
        print(f'filename: {filename}')

        s3 = boto3.client("s3")
        s3.download_file('onyx-test-001', s3_file_path, Path(save_path, filename))
        print(f'File downloaded from S3!')

        pdf_reader = PdfReader(Path(save_path, filename))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        print(f'Number of characters read from the PDF: {len(text)}')

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        print(f'Number of chunks for the selected PDF: {len(texts)}')

        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_texts(texts, embeddings,
                                     metadatas=[{"source": str(i)} for i in range(len(texts))],
                                     persist_directory=index_persist_directory + "/" + conv_id)
        # TODO: Save indices to S3 to cache the /build_indices call
        vectordb.persist()
        vectordb = None

        return jsonify({'result': 'success'})


@app.route('/get_answer_from_pdf', methods=['POST'])
def get_answer_from_pdf():
    if request.method == 'POST':
        conv_id = request.json["conversation_id"]
        query = request.json["query"]
        llm_selected = request.json["llm_selected"]

        if llm_selected != model_in_memory:
            load_model(llm_selected)

        embeddings = HuggingFaceEmbeddings()
        docsearch = Chroma(persist_directory=index_persist_directory + "/" + conv_id,
                           embedding_function=embeddings).as_retriever()

        docs = docsearch.get_relevant_documents(query)
        print(f'Number of chunks in context: {len(docs)}')

        source_docs = []
        for d in docs:
            source_docs.append(d.page_content)
        print(f'Source documents: {source_docs}')

        # TODO: to make the model type dynamic
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100,
                        device=torch.device("mps"))
        local_llm = HuggingFacePipeline(pipeline=pipe)
        chain = load_qa_chain(local_llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        print(f'Query: {query}, Answer: {response}')

        return jsonify({'result': response, 'source_docs': source_docs})


# driver function
if __name__ == '__main__':
    app.run(port=6999, debug=True)
