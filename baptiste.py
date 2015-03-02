#! /usr/bin/env python2

import sys, os
import numpy as np
from lxml import etree
import random

class Agent:
    def __init__(self, graph, start):
        assert isinstance(graph, Graph)

        self.graph = graph
        self.start = start
        self.reset()

    def reset(self):
        self.path = [self.start]
        self.path_cost = 0
        self.visited = [False]*self.graph.get_size()
        self.visited[self.start] = True

    def find_circuit(self):
        while self.pick_next_node(): pass
        self.path_cost += self.graph.get_cost(self.path[-1], self.path[0])

    def __get_edge_probabilities(self, node):
        total = 0
        probabilities = np.zeros(self.graph.get_size())
        for next_node in xrange(self.graph.get_size()):
            if next_node == node or self.visited[next_node]: continue

            raw_probability = self.graph.get_raw_probability(node, next_node)
            total += raw_probability
            probabilities[next_node] = raw_probability

        probabilities /= total
        return probabilities

    def pick_next_node(self):
        if len(self.path) >= self.graph.get_size():
            return False

        throw = random.random()
        proba = 0
        current = self.path[-1]
        probabilities = self.__get_edge_probabilities(current)
        for node in xrange(self.graph.get_size()):
            proba += probabilities[node]

            if throw < proba:
                break

        self.path.append(node)
        self.visited[node] = True
        self.path_cost += self.graph.get_cost(current, node)

        return True

class Graph:
    def __init__(self, name, size, costs):
        self.name = name
        self.size = size
        self.costs = costs
        self.pheromones = np.ones_like(costs)
        self.raw_probabilities = np.zeros_like(costs)


        for i in xrange(self.size):
            self.__update_node(i, None, 0)

    @staticmethod
    def new_from_xml(filename):
        root = etree.parse(filename)
        graph = root.find('graph')

        name = root.find('name').text
        size = len(graph.findall('vertex'))
        costs = np.full((size, size), float('infinity'))

        vertex_index = 0
        for vertex in graph.findall('vertex'):
            for edge in vertex.findall('edge'):
                costs[vertex_index, int(edge.text)] = float(edge.get('cost'))
            vertex_index += 1

        return Graph(name, size, costs)
    
    def __repr__(self):
        return self.costs.__repr__()

    def __update_node(self, start_node, chosen_node, phamount):
        if chosen_node != None:
            self.pheromones[start_node, chosen_node] += phamount

        for node in xrange(self.size):
            tau = self.pheromones[start_node, node]
            eta = self.costs[start_node, node]
            self.raw_probabilities[start_node, node] = tau * 1/eta
            
    def get_size(self):
        return self.size

    def get_raw_probability(self, start_node, next_node):
        return self.raw_probabilities[start_node, next_node]

    def get_cost(self, start_node, next_node):
        return self.costs[start_node, next_node]
    
    def swarm(self, step_limit, evap_factor, magic_phactor, agent):
        steps = 0

        while steps < step_limit:
            steps += 1
            agent.reset()
            agent.find_circuit()
            self.pheromones = (1-evap_factor)*self.pheromones
            for node in xrange(self.size-1):
                self.__update_node(agent.path[node], agent.path[node+1], magic_phactor/agent.path_cost)


    def get_best_tour(self, start): 
        current = start
        path = [current]
        total_cost = 0
        visited = [False] * self.size
        visited[start] = True

        while len(path) < self.size:
            # finds the index of the edge with the maximum amount of pheromones.
            best_node, best_phamount = None, 0
            for node, phamount in enumerate(self.pheromones[current]):
                if visited[node]: continue

                if phamount > best_phamount:
                    best_node, best_phamount = node, phamount

            total_cost += self.get_cost(current, best_node)
            current = best_node
            path.append(current)
            visited[current] = True

        total_cost += self.get_cost(path[-1], start)
        return path, total_cost

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <xml file>".format(sys.argv[0]))
        sys.exit(1)

    
    solutions = {}
    with open('problems/solutions.txt') as solutions_file:
        for line in solutions_file:
            name, cost = [item.split()[0] for item in line.split(':')]
            solutions[name] = float(cost)

    random.seed(1)
    graph = Graph.new_from_xml(sys.argv[1])
    start = 0
    agent = Agent(graph, start)
    graph.swarm(100, 0.01, 5000, agent)
    path, cost = graph.get_best_tour(start)
    best_cost = solutions[graph.name]
    print(path, cost)
    print("Error is {:.2%}.".format((cost - best_cost)/best_cost))
