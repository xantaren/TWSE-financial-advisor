import asyncio
import json

import aiohttp
import tiktoken
from bs4 import BeautifulSoup
import pandas
from io import StringIO
from openai import OpenAI
from IPython.display import Markdown
from rich.console import Console
from rich.markdown import Markdown
import google.generativeai as genai
from enum import Enum
import os
import markdown
from sqlitedict import SqliteDict
import time


class GenAiProviderEnum(Enum):
    OpenAi = 1
    GoogleGemini = 2


class GenAiProvider:
    def generate_content(self, system_instruction: str, prompt: str) -> str:
        pass


def get_provider(enum: GenAiProviderEnum) -> GenAiProvider:
    if enum == GenAiProviderEnum.OpenAi:
        return OpenAi()
    elif enum == GenAiProviderEnum.GoogleGemini:
        return GoogleGemini()


class GoogleGemini(GenAiProvider):
    def __init__(self):
        self._api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")

    def generate_content(self, system_instruction: str, prompt: str) -> str:
        api_key = self._api_key
        genai.configure(api_key=api_key)

        # model_name = 'gemini-1.5-pro'
        model_name = 'gemini-1.5-flash'

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        response = model.generate_content(prompt)
        return response.text


class OpenAi(GenAiProvider):
    def __init__(self):
        self._api_key = os.environ.get("OPEN_AI_API_KEY")

    def generate_content(self, system_instruction: str, prompt: str) -> str:
        client = OpenAI(api_key=self._api_key)
        model_name = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt},
            ],
            temperature=1,
            max_tokens=16384 if model_name == "gpt-4o-mini" else 4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content


def convert_to_string(name: str, data: BeautifulSoup) -> str:
    return f"{name}\n" + pandas.read_html(StringIO(data.prettify()))[0].to_csv()


def transform_and_filter_relevant_stock_data(json_data):
    attribute_mapping = {
        "z": "當前盤中成交價",
        "tv": "當前盤中盤成交量",
        "v": "累積成交量",
        "b": "揭示買價(從高到低，以_分隔資料)",
        "g": "揭示買量(配合b，以_分隔資料)",
        "a": "揭示賣價(從低到高，以_分隔資料)",
        "f": "揭示賣量(配合a，以_分隔資料)",
        "o": "開盤價格",
        "h": "最高價格",
        "l": "最低價格",
        "y": "昨日收盤價格",
        "u": "漲停價",
        "w": "跌停價",
        "tlong": "資料更新時間（單位：毫秒）",
        "d": "最近交易日期（YYYYMMDD）",
        "t": "最近成交時刻（HH:MI:SS）",
        "c": "股票代號",
        "n": "公司簡稱",
        "nf": "公司全名"
    }

    # Extract the array of stock data
    msg_array = json_data.get("msgArray", [])

    transformed_data = []

    for item in msg_array:
        transformed_item = {}
        for key, value in item.items():
            if key in attribute_mapping:
                # Replace the key with its Chinese counterpart
                new_key = attribute_mapping[key]
                transformed_item[new_key] = value
        transformed_data.append(transformed_item)

    return str(transformed_data)


async def make_simple_get_request(session: aiohttp.ClientSession, url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 '
                      'Safari/537.36',
        'Cookie': 'CLIENT%5FID=20240820083443203%5F220%2E134%2E120%2E14; IS_TOUCH_DEVICE=F; '
                  'SCREEN_SIZE=WIDTH=1920&HEIGHT=1080; _ga=GA1.1.1080156745.1724114085; '
                  'TW_STOCK_BROWSE_LIST=0050%7C2330%7C4171; _ga_0LP5MLQS7E=GS1.1.1724144467.5.1.1724146063.21.0.0'
    }
    async with session.get(url, headers=headers) as response:
        return await response.text(encoding='utf-8')


async def fetch_and_process_data(session: aiohttp.ClientSession, stock_number: int) -> str:
    stock_number = str(stock_number)
    twse_api = f'https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{stock_number}.tw'
    otc_api = f'https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=otc_{stock_number}.tw'

    # Look up stock assuming it's listed on Taiwan Stock Exchange (上市)
    stock_data_response = await make_simple_get_request(session, twse_api)
    stock_data = json.loads(stock_data_response)

    # If lookup failed, then try looking up on the over-the-counter (上櫃) market
    if not stock_data.get('msgArray')[0].get('pid'):
        stock_data_response = await make_simple_get_request(session, otc_api)
        stock_data = json.loads(stock_data_response)

    # If still nothing, then it's either not public or an emerging stock (興櫃) , which I don't plan on supporting
    if not stock_data.get('msgArray')[0].get('pid'):
        return ""

    basic_info = transform_and_filter_relevant_stock_data(stock_data)

    # Storing the accounting data for efficiency and to go easy on the API calls
    # The data usually takes up around 30KB per stock in the db
    # Considering there are only around 2,000 publicly traded companies in Taiwan, db size should not exceed 60MB
    with SqliteDict("db.sqlite") as db:
        # Clear data if expired
        expire_time = db.get(f'{stock_number}_expiration')
        if expire_time and time.time() > expire_time:
            db[stock_number] = None

        stored_data = db.get(stock_number)
        if stored_data:
            return f"{basic_info}\n\n{stored_data}"

        urls = [
            f'https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={stock_number}',
            f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=BS_M_QUAR&STOCK_ID={stock_number}',
            f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=IS_M_QUAR_ACC&STOCK_ID={stock_number}',
            f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=CF_M_QUAR_ACC&STOCK_ID={stock_number}',
            f'https://goodinfo.tw/tw/StockFinGrade.asp?STOCK_ID={stock_number}'
        ]

        responses = await asyncio.gather(*[make_simple_get_request(session, url) for url in urls])
        soups = [BeautifulSoup(response, 'html5lib') for response in responses]

        data = "\n\n".join(
            convert_to_string(title, soup.select_one(selector))
            for title, soup, selector in [
                ("經營績效", soups[0], '#txtFinDetailData'),
                ("資產負債表", soups[1], '#divFinDetail'),
                ("損益表", soups[2], '#divFinDetail'),
                ("現金流量表", soups[3], '#divFinDetail'),
                ("財報評比資料", soups[4], '#divDetail')
            ]
        )

        # Note that I don't store basic info because it contains current price during trading hours
        db[stock_number] = data
        db[f'{stock_number}_expiration'] = time.time() + 86400  # seconds in a day
        db.commit()

    return f"{basic_info}\n\n{data}"


def send_to_gen_ai_provider(gen_ai_provider: GenAiProviderEnum, joined_data: str):
    system_instruction = """"
    您是一位經驗豐富的價值投資者，深受華倫·巴菲特投資理念的啟發。您的主要目標是尋找具備強大基本面和長期競爭優勢但目前市值被低估的公司。
    分析重點：
    商業模式： 評估公司的商業模式是否具有可持續性、盈利能力，以及在產業中的獨特性。
    財務狀況： 深入分析公司的財務報表，重點關注以下指標：
    營收成長率、毛利率、淨利率、資產負債率、流動比率、速動比率、本益比、股價淨值比、股息殖利率、自由現金流量
    競爭優勢： 評估公司的核心競爭優勢，包括品牌、技術、成本、渠道、客戶忠誠度等。
    行業趨勢： 分析公司所處行業的發展趨勢、周期性、政策影響、以及競爭格局。
    風險分析： 評估公司可能面臨的經營風險、產業風險、政策風險等，並進行敏感性分析和情境分析。
    市場情緒： 分析市場對公司的看法，包括分析師評級、媒體報導、投資人討論等。
    評分系統：
    請根據以上分析，給予公司一個0-10分的綜合評分，並說明評分依據。
    買進建議：
    請根據公司的財務狀況、競爭優勢、行業趨勢、風險評估以及市場估值，判斷是否建議買入該公司股票，並給出建議的買入價格。
    語言要求： 所有回應均需使用台灣繁體中文，避免使用中國簡體字。
    注意事項：
    避免使用絕對詞彙： 如「絕對」、「一定」、「永遠」等。
    強調數據分析： 使用具體的數據和百分比來支持您的判斷。
    考慮長期投資： 評估公司的長期發展潛力，而非短期波動。"""

    provider = get_provider(gen_ai_provider)
    return provider.generate_content(system_instruction, joined_data)


def display_content(generated_content: str):
    console = Console()
    md = Markdown(generated_content)
    console.print(md)


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def main(stock_number: int, gen_ai_provider: GenAiProviderEnum):
    async with aiohttp.ClientSession() as session:
        data = await fetch_and_process_data(session, stock_number)

    if not data:
        return

    generated_content = send_to_gen_ai_provider(gen_ai_provider, data)
    display_content(generated_content)


if __name__ == '__main__':
    if os.name == 'nt':
        # For "RuntimeError: Event loop is closed" error  when running on Windows devices
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    start = time.time()
    asyncio.run(main(1234, GenAiProviderEnum.GoogleGemini))
    stop = time.time()

    print('Time: ', stop - start)
