import requests, csv

url = "https://3dzqiahkw1.execute-api.eu-west-1.amazonaws.com/prod/leaderboard"

def params(i):
    return {
    "type": "MANUAL",
    "page": i,
    "limit": 100
}

headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Referer": "https://prosperity.imc.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-GPC": "1",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

f = open("leaderboard_data_manual.csv", 'a')
writer = csv.writer(f)
for  i in range(222):
    response = requests.get(url, params=params(i+1), headers=headers)

    if response.status_code == 200:
        data = response.json()
        data = data['data']['items']
        for k in data:
            writer.writerow([k["position"], k["positionChange"], k["team"]["id"], k["team"]["name"], k["team"]["countryCode"], k["score"]])
    else:
        print(f"Error: {response.status_code}")
    print("Page", i+1, "done")
f.flush()
f.close()