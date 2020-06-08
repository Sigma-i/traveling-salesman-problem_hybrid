# Copyright 2020 Sigma-i Co.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

from pyqubo import Array, Placeholder, Constraint
from dwave.system import LeapHybridSampler


# --- Problem setting ---

# Define the coordinates of each city and one origin city (at random in this demo)
N = 9

X_range = Y_range = 500
x_pos = [np.random.randint(0, X_range) for _ in range(N)]
y_pos = [np.random.randint(0, Y_range) for _ in range(N)]
positions = {i: (x_pos[i], y_pos[i]) for i in range(N)}  # you can rewrite this line

# Choose origin (and end) city and fix it
origin = np.random.choice(np.arange(N))  # 
origin_pos = positions[origin]

others = list(range(N))
others.remove(origin)

# Set a graph
G = nx.Graph()
G.add_nodes_from(np.arange(N))
nx.draw(G, positions, node_color=['red' if i == origin else 'blue' for i in range(N)], with_labels=True)

# Calculate the distance between each city
distances = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        distances[i][j] = np.sqrt((x_pos[i] - x_pos[j])**2 + (y_pos[i] - y_pos[j])**2)
        distances[j][i] = distances[i][j]


# --- Problem formulation ---

# Use pyqubo package
q = Array.create('q', shape=(N-1, N-1), vartype='BINARY')

def normalize(exp):
    """ Normalization function """
    qubo, offset = exp.compile().to_qubo()
    
    max_coeff = abs(np.max(list(qubo.values())))
    min_coeff = abs(np.min(list(qubo.values())))
    norm = max_coeff if max_coeff - min_coeff > 0 else min_coeff
    
    return exp / norm

# Cost function
exp_origin = sum(distances[origin][others[i]]*1*q[i][0] + 
             distances[others[i]][origin]*q[i][N-2]*1 for i in range(N-1))
exp_others = sum(distances[others[i]][others[j]]*q[i][t]*q[j][t+1]
              for i in range(N-1) for j in range(N-1) for t in range(N-2))
H_cost = normalize(exp_origin + exp_others)

# Constraint
H_city = Constraint(normalize(sum((sum(q[i][t] for t in range(N-1))-1)**2 for i in range(N-1))), 'city')
H_time = Constraint(normalize(sum((sum(q[i][t] for i in range(N-1))-1)**2 for t in range(N-1))), 'time')

# Express objective function and compile it to model
H = H_cost + Placeholder('lam') * (H_city + H_time)
model = H.compile()


# --- Solve QUBO ---

# Get the QUBO matrix from the model
feed_dict = {'lam':5.0}  # the value of constraint
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# Run QUBO on Leap's Hybrid Solver (hybrid_v1)
sampler = LeapHybridSampler(token='') 
response = sampler.sample_qubo(qubo)
sample = response.record['sample'][0]

# decode the solution and check if constrains are satisfied
sample_dict = {idx: sample[i] for i,idx in enumerate(response.variables)}
decoded, broken, energy = model.decode_solution(sample_dict, 'BINARY', feed_dict=feed_dict)  
if broken == {}:
    print('The solution is valid')
else:
    print('The solution is invalid')


# --- Visualize the result ---

# Create an array which shows traveling order from the solution
solution = sample.reshape(N-1, N-1)
order = [origin]

for li in solution.T:
    cities = np.where(li)[0].tolist()
    if cities == []:
        continue
    if len(cities) > 1:
        order.append(others[np.random.choice(cities, 1)[0]])
    else:
        order.append(others[cities[0]])

# Plot the result
new = np.append(np.zeros((N-1,1)), solution, axis=1) 
result_arr = np.insert(new, origin, np.append(1,np.zeros(N-1)), axis=0)
    
fig = plt.figure(facecolor='w', edgecolor='k')
ax = fig.subplots()
ax.imshow(result_arr)
ax.set_xlabel('Order')
ax.set_ylabel('Cities')
plt.tight_layout()
plt.savefig('result.png')

# Draw the route on graph G   
edges = []
edges += [(order[t], order[t+1]) for t in range(len(order)-1)]
edges += [(order[-1], origin)]
    
G_copy = deepcopy(G)
G_copy.add_edges_from(edges)

plt.figure()
nx.draw(G_copy, positions, node_color=['red' if i == origin else 'blue' for i in range(N)], with_labels=True)
plt.savefig('route.png')
    
def calc_dis(order_arr):
    """Calculate total traveling distance (the value of H_cost) of the solution"""
    dis = sum(distances[order_arr[t]][order_arr[t+1]] for t in range(len(order_arr)-1))\
        + distances[order_arr[-1]][origin]
    return dis
print(f'distance: {calc_dis(order)}')