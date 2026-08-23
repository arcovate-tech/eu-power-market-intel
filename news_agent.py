import os
import json
import feedparser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a market analyst assistant. Given a news headline and summary,
extract a structured signal. Respond ONLY with valid JSON, no extra text, in this exact format:
{{"commodity": "power" or "gas" or "other", "sentiment": "bullish" or "bearish" or "neutral", "key_driver": "short phrase", "confidence": "high" or "medium" or "low"}}"""),
    ("user", "Headline: {headline}\nSummary: {summary}")
])

chain = prompt | llm

# Free energy news RSS feed
RSS_URL = "https://news.google.com/rss/search?q=European+energy+market+OR+electricity+prices+OR+natural+gas+Europe&hl=en-US&gl=US&ceid=US:en"  # placeholder, may need a working energy-specific feed

def get_articles(limit=10):
    feed = feedparser.parse(RSS_URL)
    articles = []
    for entry in feed.entries[:limit]:
        articles.append({
            "headline": entry.get("title", ""),
            "summary": entry.get("summary", "")
        })
    return articles

def extract_signal(article):
    response = chain.invoke({
        "headline": article["headline"],
        "summary": article["summary"]
    })
    try:
        signal = json.loads(response.content)
    except json.JSONDecodeError:
        signal = {"error": "could not parse", "raw_response": response.content}
    return signal

if __name__ == "__main__":
    articles = get_articles(limit=5)
    print(f"Pulled {len(articles)} articles.\n")

    for article in articles:
        print(f"Headline: {article['headline']}")
        signal = extract_signal(article)
        print(f"Signal: {json.dumps(signal, indent=2)}\n")
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def get_articles(limit=10):
    response = requests.get(RSS_URL, headers=HEADERS, timeout=10)
    feed = feedparser.parse(response.content)
    articles = []
    for entry in feed.entries[:limit]:
        articles.append({
            "headline": entry.get("title", ""),
            "summary": entry.get("summary", "")
        })
    return articles