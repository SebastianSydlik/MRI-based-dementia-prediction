import requests

patient = {

    "m/f": 1,
    "age": 99,
    "educ":2.0,
    "ses": 0.0,
    "etiv": 1500,
    "nwbv": 0.1,    
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient)

print(response.json())
