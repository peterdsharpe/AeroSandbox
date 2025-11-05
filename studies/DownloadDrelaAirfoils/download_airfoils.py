import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0",
}


url = "http://www.charlesriverrc.org/articles/drela-airfoilshop/markdrela-ag-ht-airfoils.htm"
url_base = "http://www.charlesriverrc.org/articles/drela-airfoilshop/"
folder = Path(__file__).parent / "airfoils"

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

for link in soup.find_all("a"):
    href = link.get("href")
    if href.endswith(".dat"):
        # get filename from link
        filename = os.path.basename(href)
        # download file
        r = requests.get(url_base + href, headers=headers)
        with open(folder / filename, "wb") as f:
            f.write(r.content)
            print(f"{filename} downloaded.")
