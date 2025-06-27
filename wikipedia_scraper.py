import wikipediaapi
import pandas as pd
from tqdm import tqdm

# Setup Wikipedia API
wiki_en = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MultilingualSearchBot/1.0 (praveen.verma81k@gmail.com)'  
)
wiki_hi = wikipediaapi.Wikipedia(
    language='hi',
    user_agent='MultilingualSearchBot/1.0 (praveen.verma81k@gmail.com)'  
)

# List of common topics that exist in both languages
topics = [
    # Cricket/Sports
    "Virat Kohli", "MS Dhoni", "ICC Cricket World Cup", "Sachin Tendulkar", "Cricket in India",

    # News / Politics / Economy
    "Lok Sabha elections", "Union Budget of India", "India-Pakistan relations", "Inflation", "Ayodhya Ram Mandir",

    # Entrance Exams / Education
    "IIT JEE", "NEET", "UPSC", "CBSE", "Kendriya Vidyalaya",

    # Tech / Science
    "Machine learning", "Artificial intelligence", "ISRO", "Chandrayaan-3", "Digital India",

    # Health
    "COVID-19", "Diabetes", "Yoga", "Nutrition", "Pollution in Delhi"
]

data = []

for topic in tqdm(topics):
    page_en = wiki_en.page(topic)
    page_hi = wiki_hi.page(topic)

    if page_en.exists() and page_hi.exists():
        data.append({"topic": topic, "language": "English", "text": page_en.summary})
        data.append({"topic": topic, "language": "Hindi", "text": page_hi.summary})
    else:
        print(f"Missing one of the pages for: {topic}")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(r"C:\Users\PRAVEEN\OneDrive\Desktop\Multilingual Semantic Search Engine/multilingual.csv", index=False, encoding='utf-8-sig')
print(" Data saved to app/data/multilingual.csv")
