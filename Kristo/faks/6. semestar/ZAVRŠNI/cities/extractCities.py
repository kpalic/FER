import csv
import json
import gzip

# Učitavanje podataka iz JSON datoteke
with gzip.open(r"C:\Users\eaprlik\Desktop\ZAVRŠNI\cities\city.list.json.gz", "r") as f:
    data = f.read()
    json_data = json.loads(data.decode('utf-8'))

# Definiranje naziva kolona za CSV datoteku
fieldnames = ['id', 'name', 'state', 'country', 'lon', 'lat']

# Pisanje podataka u CSV datoteku
with open('gradovi.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in json_data:
        flat_row = {
            'id': row['id'],
            'name': row['name'],
            'state': row['state'],
            'country': row['country'],
            'lon': row['coord']['lon'],
            'lat': row['coord']['lat'],
        }
        writer.writerow(flat_row)
