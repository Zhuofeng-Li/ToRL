import requests
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    url = 'http://192.168.0.86:8080/run_code' if args.remote else 'http://0.0.0.0:8080/run_code'
    response = requests.post(url, json={
        'code': "import sympy as sp\n\n# Define the coordinates of points A and C\nA = sp.Point(0, sp.sqrt(10))\nC = sp.Point(2*sp.sqrt(10), sp.sqrt(10))\n\n# Calculate the distance between A and C\nAC = A.distance(C)\nprint(AC)",
    'language': 'python',
})
    print(json.dumps(response.json(), indent=2))