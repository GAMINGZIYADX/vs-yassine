from bs4 import BeautifulSoup
import requests
import csv

url = "https://books.toscrape.com/index.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

books = soup.find_all("article", class_="product_pod")

data = []  

for book in books:
    title = book.h3.a["title"]
    price = book.find("p", class_="price_color").get_text(strip=True)
    stars = book.find("p", class_="star-rating")["class"][1]
    available = book.find("p", class_="instock").get_text(strip=True)
    link = book.h3.a["href"]
    
    data.append({
        "title": title,
        "price": price,
        "stars": stars,
        "availability": available,
        "link": link
    })


with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "stars", "availability", "link"])
    writer.writeheader()
    writer.writerows(data)

print("Saved to books.csv")

