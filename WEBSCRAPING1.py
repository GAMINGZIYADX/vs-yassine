
from bs4 import soup

    
results = soup.find_all("div", class_="item-result")

for item in results:
    h3 = item.find("h3")
    title = h3.get_text(strip=True) if h3 else "N/A"

    desc_div = item.find("div", class_="nl-wrap")
    raw = desc_div.get_text(strip=True) if desc_div else "N/A"
    description = raw.replace("<p>", "").replace("</p>", "")

    print(title)
    print(description)
    print("-" * 40)