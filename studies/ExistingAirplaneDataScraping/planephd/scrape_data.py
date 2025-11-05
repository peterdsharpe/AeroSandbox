import requests
from bs4 import BeautifulSoup
import re

# The main URL that contains links to all airplanes
main_url = "https://planephd.com/wizard/?modeltype=all&annual_hrs=100&min_speed=0&purchase_price_max=100000000000&ownership_cost_p_year_max=100000000000&required_seats_min=0&min_year=1900"

# Send a GET request to the main URL
response = requests.get(main_url)

# Parse the response text with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Find all the individual airplane model links
airplane_links = [
    a["href"] for a in soup.find_all("a", href=re.compile(r"^/wizard/details"))
]

# Initialize an empty dictionary to store the airplane data
airplanes_data = {}

# For each airplane link
for link in airplane_links:
    # Construct the full URL
    airplane_url = "https://planephd.com" + link

    # Send a GET request to the airplane URL
    response = requests.get(airplane_url)

    # Parse the response text with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the airplane model name
    model_name = soup.find("div", class_="model-header").h3.text.strip()

    # Initialize an empty dictionary to store the airplane model data
    airplanes_data[model_name] = {}

    # Find the Performance Specifications and Weights tables
    tables = soup.find_all("div", class_="col-md-12")

    # For each table
    for table in tables:
        # Find all the rows in the table
        rows = table.find_all("dl", class_="dl-horizontal dl-details dl-skinny")

        # For each row
        for row in rows:
            # Find all the columns in the row
            cols = row.find_all(["dt", "dd"])

            # If there are two columns
            if len(cols) % 2 == 0:
                # The first column is the key and the second column is the value
                for i in range(0, len(cols), 2):
                    key = cols[i].text.strip()
                    value = cols[i + 1].text.strip()

                    # Add the key-value pair to the airplane model data
                    airplanes_data[model_name][key] = value

    # Print the airplane name and data
    print(model_name)
    for key, value in airplanes_data[model_name].items():
        print(f"{key}: {value}")
    print()

# Print the airplanes data
for model_name, model_data in airplanes_data.items():
    print(model_name)
    for key, value in model_data.items():
        print(f"{key}: {value}")
    print()
