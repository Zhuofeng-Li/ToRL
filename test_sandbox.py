import requests
import json

response = requests.post('http://0.0.0.0:8080/run_code', json={
    'code': "import sympy as sp\n\n# Define the variables\nx1, y1, x2, y2, x3, y3 = sp.symbols('x1 y1 x2 y2 x3 y3')\n\n# Define the equations\neq1 = (x1 - x2)**2 + (y1 - y2)**2 - 9\neq2 = (x1 - x3)**2 + (y1 - y3)**2 - 16\neq3 = (x2 - x3)**2 + (y2 - y3)**2 - 25\n\n# Define the determinant\ndet = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)\n\n# Solve the equations for the squared determinant\nsolution = sp.solve([eq1, eq2, eq3], (x1, y1, x2, y2, x3, y3), dict=True)\n\n# Calculate the squared determinant for one of the solutions\ndet_squared = det.subs(solution[0])**2\n\nprint(det_squared)\n",
    'language': 'python',
})

print(json.dumps(response.json(), indent=2))