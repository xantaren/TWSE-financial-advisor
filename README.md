## Overview
This script fetches real-time stock data for Taiwanese companies from multiple sources, processes it, and then generates an AI-driven financial analysis of the stock using either OpenAI or Google Gemini (Generative AI). The analysis focuses on key financial indicators and value investing principles.

## Features
Fetches and processes stock data from Taiwan Stock Exchange (TWSE) and Over-the-Counter (OTC) markets.
Utilizes BeautifulSoup for web scraping and aiohttp for asynchronous HTTP requests.
Transforms raw stock data into a structured format using a mapping of important financial attributes.
Integrates with two AI content generation platforms, OpenAI and Google Gemini, for automated financial analysis based on value investing principles.
Stores data in an SQLite database to reduce redundant API calls and improve performance.
Supports both short-term (current stock prices) and long-term (financial report data) analysis.

## Requirements
Python 3.9+
Required packages (install via pip):
```
pip install aiohttp beautifulsoup4 pandas tiktoken openai rich google-generativeai sqlitedict markdown
```

## Environment Variables
You need to set API keys for the AI content generation services:

- `OPEN_AI_API_KEY`: API key for OpenAI.
- `GOOGLE_GEMINI_API_KEY`: API key for Google Gemini.

## Example Workflow:
- Fetch Stock Data: Retrieves stock data (price, volume, buy/sell information, etc.) from public APIs and stock financial reports from a financial website.

- Store in Database: If the data is new or the cache has expired, it stores the fetched data in an SQLite database for 24 hours to minimize API requests.

- AI-Driven Analysis: Passes the data to either OpenAI or Google Gemini for AI-driven analysis, generating a financial report based on key metrics like revenue, net income, and competitive advantages.

- Display Analysis: The generated analysis is displayed in Markdown format using the rich library for enhanced console output.

## Parameters
- `stock_number`: The numeric code of the stock to fetch.
- `gen_ai_provider`: Choose between GenAiProviderEnum.OpenAi or GenAiProviderEnum.GoogleGemini for the AI analysis provider.
Output
Stock financial data is fetched, processed, and displayed in a detailed, AI-generated financial analysis with a final investment score and recommendations.

## Customization
- API Providers: You can switch between OpenAI and Google Gemini by modifying the gen_ai_provider parameter in the main() function.
- Database Expiration: The cache expiration is set to 24 hours by default. Modify db[f'{stock_number}_expiration'] for custom expiration times.
