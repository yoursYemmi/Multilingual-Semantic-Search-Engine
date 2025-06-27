import feedparser
import pandas as pd

def scrape_rss(feed_url, topic, language):
    print(f" Scraping: {language.upper()} - {topic}")
    entries = []
    feed = feedparser.parse(feed_url)

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        if title and summary:
            entries.append({
                "topic": topic,
                "language": language,
                "text": f"{title} - {summary}"
            })

    return entries


# Jagran Hindi RSS Feeds
jagran_rss = {
    "Education": "https://www.jagran.com/rss/education.xml",
    "Politics": "https://www.jagran.com/rss/politics.xml",
    "Science": "https://www.jagran.com/rss/science.xml"
}

#  Live Hindustan Hindi RSS Feeds
live_hindustan_rss = {
    "India": "https://www.livehindustan.com/rss/national.xml",
    "Business": "https://www.livehindustan.com/rss/business.xml",
    "Education": "https://www.livehindustan.com/rss/education.xml"
}

#  The Hindu English RSS Feeds
the_hindu_rss = {
    "Education": "https://www.thehindu.com/education/feeder/default.rss",
    "Technology": "https://www.thehindu.com/sci-tech/technology/feeder/default.rss",
    "National": "https://www.thehindu.com/news/national/feeder/default.rss"
}


#Combine all
all_articles = []

# Jagran
for topic, url in jagran_rss.items():
    all_articles.extend(scrape_rss(url, topic, "Hindi"))

# Live Hindustan
for topic, url in live_hindustan_rss.items():
    all_articles.extend(scrape_rss(url, topic, "Hindi"))

# The Hindu
for topic, url in the_hindu_rss.items():
    all_articles.extend(scrape_rss(url, topic, "English"))

# Save as CSV
df = pd.DataFrame(all_articles)
df.to_csv(r"C:\Users\PRAVEEN\OneDrive\Desktop\Multilingual Semantic Search Engine/news_combined_rss.csv", index=False, encoding="utf-8-sig")
print(f"\n Saved {len(df)} articles to app/data/news_combined_rss.csv")
