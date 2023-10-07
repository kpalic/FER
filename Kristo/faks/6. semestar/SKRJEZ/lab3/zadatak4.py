import urllib.request
import re
from urllib.parse import urlparse

def print_web_info(url):
    # Otvorimo i učitamo zadanu web stranicu
    stranica = urllib.request.urlopen(url)
    mybytes = stranica.read()
    mystr = mybytes.decode("utf8")

    # Ispisujemo sadržaj web stranice
    print(mystr)

    # Pronađemo i izlistamo sve linkove na druge stranice
    links = re.findall('href="([^"]*)"', mystr)
    print("\nLinkovi na druge stranice:")
    for link in links:
        print(link)

    # Napravimo listu svih hostova kojima se sa stranice može pristupiti (bez ponavljanja)
    hosts = set(urlparse(link).netloc for link in links)
    print("\nHostovi kojima se sa stranice može pristupiti:")
    for host in hosts:
        print(host)

    # Za svaki host odredimo broj referenciranja u razmatranoj stranici
    print("\nBroj referenciranja po hostovima:")
    for host in hosts:
        count = sum(urlparse(link).netloc == host for link in links)
        print(f"{host}: {count}")

    # Pronađemo sve e-mail adrese u toj stranici
    emails = re.findall('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', mystr)
    print("\nEmail adrese na stranici:")
    for email in emails:
        print(email)

    # Prebrojimo linkove na slike
    images = re.findall('<img src="([^"]*)"', mystr)
    print(f"\nBroj linkova na slike: {len(images)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Molim Vas unesite URL kao argument naredbenog retka!")
    else:
        print_web_info(sys.argv[1])
