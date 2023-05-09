from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BloomTokenizerFast, \
    BloomForCausalLM

# Creating a Flask app
app = Flask(__name__)


# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"
model_in_memory = ""

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

        path = "bigscience/bloomz-560m"
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
        model = BloomForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
        # Set the below parameter for low RAM usage
        # model = BloomForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
    elif "vicuna" in path:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, low_cpu_mem_usage=True)
    model.to(device)
    print(f'Loading model finished...')


# TODO: implement llm_save method functionality to handle other model types
@app.route('/llm_save', methods=['POST'])
def llm_save():
    if request.method == 'POST':
        print(f'Saving model started...')
        save_path = "/Users/snehalyelmati/Documents/models/"
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


# TODO: replace all request.form to request.data
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


# driver function
if __name__ == '__main__':
    app.run(port=5999, debug=True)
