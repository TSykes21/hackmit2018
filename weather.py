import requests
headers1 = {"accept": "application/json; charset=UTF-8"}
headers2 = {"accept": "application/json; Accept-Encoding: gzip"}
headers3 = {"accept": "application/json; charset=UTF-8"}
headers4 = {"accept": "application/json; charset=UTF-8"}

response1 = requests.get("https://api.weather.com/v3/wx/forecast/daily/7day?geocode=12.97%2C77.5&units=e&language=en-US&format=json&apiKey=320c9252a6e642f38c9252a6e682f3c6", headers=headers1)
#response2 = requests.get("https://api.weather.com/v2/stormreports?geoId=42.15%2C-71.05&format=json&apiKey=320c9252a6e642f38c9252a6e682f3c6", headers=headers2)

#Currents on demand
response3 =requests.get("https://api.weather.com/v3/wx/observations/current?geocode=42.15%2C-71.05&units=e&language=en-US&format=json&apiKey=320c9252a6e642f38c9252a6e682f3c6", headers=headers3)

#
response4 = requests.get("https://api.weather.com/v3/wx/observations/current?geocode=42.15%2C-71.05&units=e&language=en-US&format=json&apiKey=320c9252a6e642f38c9252a6e682f3c6", headers=headers4)

response1_json = response1.json()
#response2_json = response2.json()
response3_json= response3.json()
response4_json =response4.json()
#print(response.json()['dayOfWeek'])

li1 = []
for key in response1_json:
    li1.append(response1_json[key][0])

print(li1)

'''
li2 = []
for key in response2_json:
    li2.append(response2_json[key])

if len(li2) == 0:
    print("Empty")
else:
    print(li2)
'''

li3 = []
for key in response3_json:
    li3.append(response3_json[key])

#print(li3)

li4 = []
for key in response4_json:
    li4.append(response4_json[key])

#print(li4)