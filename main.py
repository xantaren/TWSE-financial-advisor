import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas
from io import StringIO
from openai import OpenAI
from IPython.display import Markdown
from rich.console import Console
from rich.markdown import Markdown
import google.generativeai as genai
from enum import Enum


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
        self._api_key = "Not today!"

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
        self._api_key = "Not today!"

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
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content


def convert_to_string(name: str, data: BeautifulSoup) -> str:
    return f"{name}\n" + pandas.read_html(StringIO(data.prettify()))[0].to_string()


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


async def fetch_and_process_data(session: aiohttp.ClientSession, stock_number: int) -> list:
    urls = [
        f'https://goodinfo.tw/tw/BasicInfo.asp?STOCK_ID={stock_number}',
        f'https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={stock_number}',
        f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=BS_M_QUAR&STOCK_ID={stock_number}',
        f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=BS_M_QUAR&STOCK_ID={stock_number}',
        f'https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=CF_M_QUAR_ACC&STOCK_ID={stock_number}'
    ]

    tasks = [make_simple_get_request(session, url) for url in urls]
    responses = await asyncio.gather(*tasks)

    soups = [BeautifulSoup(response, 'html5lib') for response in responses]

    data = [
        convert_to_string("基本資料表", soups[0].find('table', class_='b1 p4_6 r10 box_shadow')),
        convert_to_string("財務報表", soups[1].select_one('#txtFinDetailData')),
        convert_to_string("資產負債表", soups[2].select_one('#divFinDetail')),
        convert_to_string("損益表", soups[3].select_one('#divFinDetail')),
        convert_to_string("現金流量表", soups[4].select_one('#divFinDetail'))
    ]

    return data


async def main(stock_number: int, gen_ai_provider: GenAiProviderEnum):
    async with aiohttp.ClientSession() as session:
        data = await fetch_and_process_data(session, stock_number)

    joined_data = "\n\n".join(data)
    print(joined_data)

    system_instruction = """您是一位經驗豐富的投資者，效仿華倫‧巴菲特。
    您的主要目標是識別具有強大基本面和長期競爭優勢的被低估的公司。
    考慮商業模式、管理品質、財務健康狀況和行業趨勢等因素。
    提供潛在投資機會的詳細分析，包括風險和回報。你使用的語言為台灣繁體中文，請勿在回覆中夾帶中國簡體字。
    請你以長期價值投資的角度，對以下公司的財報、資產負債表、損益表、現金流量表進行深入分析。最後以滿分十分最低零分給予綜合分數評價"""

    provider = get_provider(gen_ai_provider)
    generated_content = provider.generate_content(system_instruction, joined_data)

    console = Console()
    md = Markdown(generated_content)
    console.print(md)


if __name__ == '__main__':
    asyncio.run(main(3663, GenAiProviderEnum.OpenAi))
