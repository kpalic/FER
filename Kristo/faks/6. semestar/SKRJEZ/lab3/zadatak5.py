import sys

def main():
    if len(sys.argv) != 2:
        print('Korištenje: py zadatak5.py [ime_datoteke]')
        print('Neispravan broj argumenata')
        return
    parse_file(sys.argv[1])

def parse_file(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # print(lines)
    
    # Skripta treba provjeriti uspješnost otvaranja navedene datoteke s podacima. U slučaju
    # neispravnog pozivanja ispisati uputu o načinu korištenja i izaći iz skripte. 
    except FileNotFoundError:           
        print(f"Nije moguće otvoriti datoteku: {file_name}")
        return

    print('<UL>')
    for line in lines:
        line = line.strip()
        if line:
            surnameName, title, year = line.split(';')
            surname, name = surnameName.split(',')
            print(f'    <LI> {name} {surname}, <I>{title}</I>, {year} </LI>')
    print('</UL>')


if __name__ == "__main__":
    main()
