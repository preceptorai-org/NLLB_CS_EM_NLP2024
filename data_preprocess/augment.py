import csv
from tqdm import tqdm
import random
import asyncio
import time

from pathlib import Path

# from openai import OpenAI
import json
import requests
import httpx

# Read API Keys
API_PATH = Path("./api.json")
with open(API_PATH, "r", encoding="utf-8") as f:
    API_KEYS = json.loads(f.read())

GEMINI_KEY = API_KEYS["gemini"]

SRC_CSV_PATH = "./translation_pairs_2022-08-19.csv"
DEST_CSV_PATH = "./translation_pairs_2022-08-19_augmented.csv"

data = []

# Load if file exists else just dont load from the last file
if Path(DEST_CSV_PATH).exists():
    print("Load from snapshot", DEST_CSV_PATH)
    with open(DEST_CSV_PATH, "r", encoding="utf-8-sig") as f:
        data = list(csv.DictReader(f))
else:
    print("Load from Datasrouce", SRC_CSV_PATH)
    with open(SRC_CSV_PATH, "r", encoding="utf-8-sig") as f:
        data = list(csv.DictReader(f))
    data = [item for item in data if item["response_th"].strip() != ""]
    # set  all data to same schema
    for item in data:
        item["gemini_can_augment"] = ""
        item["gemini_target_text"] = ""


def save_snapshot():
    print("Save sanpe shot to ", DEST_CSV_PATH)
    with open(DEST_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for item in tqdm(data):
            writer.writerow(item)


def gemini_serial(text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_KEY}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "contents": [{"parts": [{"text": text}]}],
                "safetySettings": [
                    {"category": category, "threshold": "BLOCK_NONE"}
                    for category in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    ]
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "topP": 1.0,
                    "topK": 1,  # definitely change this if we gonna make like more diverse stuff
                },
            }
        ),
    )
    response = resp.json()
    print(response)
    return response["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_async(client: httpx.AsyncClient, text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_KEY}"
    resp = await client.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "contents": [{"parts": [{"text": text}]}],
                "safetySettings": [
                    {"category": category, "threshold": "BLOCK_NONE"}
                    for category in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    ]
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "topP": 1.0,
                    "topK": 1,  # definitely change this if we gonna make like more diverse stuff
                },
            }
        ),
        timeout=None,
    )
    response = resp.json()
    if (
        "candidates" in response
        and len(response["candidates"]) == 1
        and response["candidates"][0]["finishReason"] == "STOP"
    ):
        return True, response["candidates"][0]["content"]["parts"][0]["text"]
    else:
        print(response)
        return False, ""


from langchain.prompts import PromptTemplate


augment_prompt = r'''
You are a medical professional. Please reformulate the sentence. Retain Thai language used while English words that appear to be jargons or are likely to be uncommon in Thai language should be in English.

Text: """
{text}
"""

Reformulation: 
'''

back_translate_prompt = r'''
Translate the following text to English in medical context:

Text: """
{text}
"""

Translation:
'''


augment_prompt = PromptTemplate.from_template(augment_prompt)
back_translate_prompt = PromptTemplate.from_template(back_translate_prompt)


def async_main():

    async def main():
        data_to_request = [
            (i, item) for i, item in enumerate(data) if item["gemini_can_augment"] == ""
        ]

        async with httpx.AsyncClient() as client:

            async def process_item(item, idx):
                print("Gemini Augment", idx)
                user_message = augment_prompt.format(text=item["response_th"])
                can_augment, response = await gemini_async(client, user_message)
                item["gemini_can_augment"] = can_augment
                item["gemini_target_text"] = response
                print("Finish processing item", idx)

            REQ_PER_WINDOW = 55
            SAVE_EVERY = REQ_PER_WINDOW * 5
            for i in tqdm(range(0, len(data_to_request), REQ_PER_WINDOW)):
                start_time = time.time()

                tasks = [
                    process_item(data_to_request[j][1], data_to_request[j][0])
                    for j in range(i, i + REQ_PER_WINDOW)
                    if j < len(data_to_request)
                ]
                await asyncio.gather(*tasks)
                if i % SAVE_EVERY == 0:
                    save_snapshot()

                end_time = time.time()

                # wait for RPM
                duration = end_time - start_time
                duration_needed_to_wait = 62 - duration
                if duration > 0:
                    print("waiting for RPM in ", duration_needed_to_wait, "seconds")
                    time.sleep(duration_needed_to_wait)

    asyncio.run(main())


def serial_main():
    idx_list = [i for i in range(len(data))]
    random.shuffle(idx_list)
    # random.shuffle(data)
    for idx in tqdm(idx_list):
        item = data[idx]

        if item["response_th"].strip() != "":

            print("Gemini Augment", idx)
            user_message = augment_prompt.format(text=item["response_th"])
            response = gemini_serial(user_message)
            item["gemini_target_text"] = response

            print("Gemini Back Translate", idx)
            user_message = back_translate_prompt.format(text=item["response_th"])
            response = gemini_serial(user_message)
            item["gemini_text"] = response

        print("Finish processing item", idx)
        # pprint(item)
        for k, v in item.items():
            print(k, ":", v)
        # print(item["question"],"\n",item["llm_raw_response"], "\n", item["answer"])
        input("")


# serial_main()
async_main()

save_snapshot()
