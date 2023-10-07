import time
import requests

import requests

url = 'https://ctf.zsis.hr/'
response = requests.get(url)

date = response.headers.get('Date')
print(date)

unix_time = int(time.mktime(time.strptime(date, '%a, %d %b %Y %H:%M:%S GMT')))
print(unix_time)

url = f"https://ctf.zsis.hr/challenges/2_programming_time.php/?answer={date}"
print(url)
requests.get(url)

print(response.text)  # print the response content (e.g. "Correct!" or "Incorrect!")
