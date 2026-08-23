import requests
import feedparser

RSS_URL = "https://www.spglobal.com/energy/en/rss/blog"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

response = requests.get(RSS_URL, headers=HEADERS, timeout=10)
print("HTTP status code:", response.status_code)
print("First 500 chars of response:")
print(response.text[:500])

feed = feedparser.parse(response.content)
print("\nBozo:", feed.bozo)
print("Number of entries:", len(feed.entries))