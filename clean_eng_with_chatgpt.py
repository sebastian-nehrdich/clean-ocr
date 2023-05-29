import fasttext
import pandas as pd
import multiprocessing
import os
import faiss
import requests
import time
import json
import chinese_converter as cc
import openai
import random
import re
from tqdm import tqdm 

data_path = "./txt/"

# enter your openai API key here
openai.api_key = ""


CLEANUP_PROMPT = """
Act as a text-cleaning pipeline. Strictly follow the cleaning steps below. Your input text is delimited by <>.

Cleaning Steps:
1 - remove headlines.
2 - remove extra spaces.
3 - remove only brackets and numbers.
4 - fix spelling errors.
5 - split text into sentences.
6 - join sentence which are split over multiple lines.

Output each sentence on a new line.
Do not report your steps and progress.

Input Text:
<{}>
"""



def construct_prompt(input_segment):
    prompt = CLEANUP_PROMPT.format(input_segment)
    return prompt

def get_cleaned_text(input_sentence):
    prompt = construct_prompt(input_sentence)
    message_log = [
        {"role": "user", "content": prompt}
    ]
    # need to loop here as we don't know if we get a response or a timeout
    while True:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_log,
            max_tokens=2000,
            temperature=0.1,
            #top_p=1,
            )        
            response = response.choices[0].message.content
            return response
        # in case openai API times out, retry after 5 seconds
        except Exception as e:            
            print(e)
            time.sleep(5)
            print("retrying after timeout")
            continue

def preprocess_segment(segment):
    segment = re.sub("\n", " ", segment)
    segment = re.sub(" +", " ", segment)
    return segment

def postprocess_segment(segment):        
    segment = re.sub("([\.|\?|!|;])(?![\"|\'])", "\\1\n", segment)    
    segment = re.sub("(?<=[\.|\?|!|;])[\"]", "\"\n", segment)
    segment = re.sub("(?<=[\.|\?|!|;])[\']", "\'\n", segment)
    segment = re.sub("\n +", "\n", segment)
    segment = re.sub("^[^a-zA-Z]*$", "", segment)    
    return segment


def clean_file(path):
    """
    Processes an OCR file and cleans it up with the help of ChatGPT. does some additional regex-based postcorrecting of the output.
    """
    if not "cleaned" in path:
        print("cleaning file: ", path)
        cfile = open(path, "r")
        lines = cfile.readlines()
        current_segment = ""
        cleaned_result = ""
        segments = []
        for line in tqdm(lines):
            current_segment = current_segment+line
            if len(current_segment) > 1000:
                current_segment = preprocess_segment(current_segment)
                cleaned_result += get_cleaned_text(postprocess_segment(current_segment))
                current_segment = ""
                
        with open(path[:-4]+"_cleaned.txt", "w") as file:
            file.write(cleaned_result)

# put the example files to be cleaned in ./txt/
data_path = "./txt/"
if __name__ == "__main__":
    files = os.listdir(data_path)
    files = [data_path+file for file in files]    
    pool = multiprocessing.Pool(processes=8)
    pool.map(clean_file, files)
    pool.close()
    
