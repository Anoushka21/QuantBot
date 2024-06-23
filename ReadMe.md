# QuantBot

**Objective:** Build a chatbot which can answer user's queries about their assets and portfolio data, perform calculations and give insights.

**Design**

![alt text](<img\Design.png>)
The Chatbot takes user's query and sends the query and database schema to the open source LLM which generates an SQL Query. This SQl Query is run and the results are sent back to the LLM which parses it and gives the answer in natural language that is easily understood by anyone.
<br>

**TechStack**

Main techstack includes-

![alt text](<img\techstack.png>)

## Installation

Set up MySQL database called quantbot.Create tables and ingest sample data into it using-

```python
python src\db.py 
```

Install packages
```
python -m venv venv
pip install -r requirements.txt 
```
Add .env and Hugging Face Token or GROQ API KEY to get access to LLM

```
HUGGINGFACEHUB_API_TOKEN = xxx
GROQ_API_KEY = xxx
```

Launch the App 

``` streamlit run src\app.py```

## Notes

- Currently using the GROQ API with the mixtral-8x7b-32768 model because I couldn't set up a local LLM and test it due to RAM issues. We can easily replace this by using the ChatHuggingFace Chat Models in Langchain and directly pick a LLM from Hugging Face Hub. This would ensure that there is no associated cost.

```python
# Use HuggingFacePipeline instead of ChatGroq

# llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

llm = HuggingFacePipeline.from_model_id(
    model_id="xxxx",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

```

- The ChatBot doesn't answer all questions well right now mainly because the tables have data columns which are JSON fields. These columns have a lot of keys and the LLM doesn't have context on these exact keys. If we provide the LLM with the exact structure and meaning of these keys as context it improve its answering capability. Prompt engineering the template with context would help.

- Another way to improve could be finetuning the LLM itself to perform better on Querying task or expirement with other open source LLM's available.
