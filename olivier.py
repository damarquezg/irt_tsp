#!/usr/bin/env python


from lxml import etree
import sys, os
import random
import yaml
from time import time
import pylab
from profilestats import profile
from heapq import nsmallest
from multiprocessing import Process, Queue

pylab.ion()

class Node:
    '''
    A node is a town
    It has a cost to travel to other towns
    '''
    def __init__(self, xml_vertex):
        self.cost_to = {}
        for edge in xml_vertex.findall('edge'):
            self.cost_to[int(edge.text)] = int(float(edge.get('cost')))
                      
class Indiv:
    '''
    An individual is a given ordering of towns
    The cost of an individual is the cost to do a round trip
    '''
    def __init__(self, ordering):
        self.ordering = [0] + ordering
    def compute_cost(self):
        self.cost = sum([self.nodes[self.ordering[i-1]].cost_to[self.ordering[i]] for i in xrange(self.n)])

    def mutation(self):
        '''
        Performs a mutation of this individual
        Switches two nodes in the ordering
        '''
        while random.random() < 0.1:
            n1 = random.randint(1,self.n-1)
            n2 = n1
            while n1 == n2:
                n2 = random.randint(1,self.n-1)
            v = self.ordering[n1]
            self.ordering[n1] = self.ordering[n2]
            self.ordering[n2] = v
        # compute cost
        self.compute_cost()
        
    def cross(self, other):
        '''
        Performs a crossing between this and the other
        Returns a new individual with its cost computed
        '''
        # random index
        #print 'self', self.ordering
        #print 'other', other.ordering
        n = random.randint(2,self.n-2)
        #print 'n', n
        # keep the start of this 
        ordering = self.ordering[1:n]
        # store the values that are before the index in other 
        idx = other.ordering.index(ordering[-1])
        #print 'idx', idx
        beginning = [i for i in other.ordering[1:idx] if i not in ordering]
        # append ending of other
        ordering += [i for i in other.ordering[idx:] if i not in ordering]
        # append beginning of other
        ordering += beginning
        return Indiv(ordering)# no need to compute cost as crossings are mutated
    
def find_n_min_index(l, n):
    i = l.index(min(l))
    if n == 1:
        return [i]
    if i == 0:
        return [i] + find_n_min_index(l[1:], n-1)
    elif i == len(l)-1:
        return [i] + find_n_min_index(l[:-1], n-1)
    return [i] + find_n_min_index(l[:i-1]+l[i+1:],n-1)
        
        
@profile()        
def ga_min(population, config, q = None):
    '''
    Starting from the initial population, uses genetic algorithm to build an optimal individual
    '''
    # parse config
    t = time()
    max_iter = config['max_iter']               # maximum of iterations
    iter_out = config['iter_out']               # exits if the best is the same for iter_out iterations
    keep_best = config['keep_best']             # elitism
    selection_type = config['selection_type']   # selection process at the beginning of a new iteration
    
    # init global values
    pop_size = len(population)
    half_pop = pop_size/2
    total_rank = pop_size*(pop_size+1)/2
   
    # compute cost of initial individuals
    for indiv in population:
        indiv.compute_cost()
    costs = [indiv.cost for indiv in population]
    nbest = nsmallest(keep_best, costs)
    best_costs = []
    best_cost = nbest[0]
    
    it = 0
    iter_follow = 0
    while it < max_iter and iter_follow < iter_out:
        it += 1
        
        # elitism: build new population from best individuals
        new_pop = [population[costs.index(best_cost)] for best_cost in nbest]
            
        # selection
        # 1 vs 1 tournaments to fill up half the new population from the previous one
        #competitors = []
        for i in xrange(half_pop - keep_best):
            n1 = random.randint(0,pop_size-1)
            #while n1 in competitors:
            #    n1 = random.randint(0,pop_size-1)
            #competitors.append(n1)
            n2 = random.randint(0,pop_size-1)
            #while n2 in competitors or n2 == n1:
            #    n2 = random.randint(0,pop_size-1)
            #competitors.append(n2)
            if population[n1].cost < population[n2].cost:
                new_pop.append(population[n1])
            else:
                new_pop.append(population[n2])  
        
        # reproduction
        for i in xrange(half_pop):
            n1 = random.randint(0,half_pop-1)
            n2 = random.randint(0,half_pop-1)
            new_pop.append(new_pop[n1].cross(new_pop[n2]))
        # mutation
        for indiv in new_pop:
            indiv.mutation()
           
        # change to population
        population = new_pop
        costs = [indiv.cost for indiv in population]
        nbest = nsmallest(keep_best, costs)
        best_idx = costs.index(nbest[0])
        
        # check for best indiv
        if best_cost > population[best_idx].cost:
            # new best individual, save it 
            best_cost = population[best_idx].cost
            iter_follow = 0
        else:
            iter_follow += 1
        best_costs.append(best_cost)
        
    print '  Found best cost', best_cost, 'in', it, 'iterations and', round((time()-t), 2), 's'
    if q != None:
        q.put([population[best_idx], best_costs])
    else:
        return population[best_idx], best_costs
    
        
   
    
    


if __name__ == '__main__':

    # check for input file
    if len(sys.argv) < 1:
        print 'Give a sys.argv[1] file'
        sys.exit(0)
    if not os.path.lexists(sys.argv[1]):
        print 'File', sys.argv[1], 'does not exist'
        sys.exit(0)
        
    try:
        with open(sys.argv[1]) as f:
            xml = etree.fromstring(f.read())    # root element of xml
        # get all vertices - this should also check the xml syntax
        vertices = xml.find('graph').findall('vertex')
    except:
        print 'Could not load', sys.argv[1]
        sys.exit(0)
        
    # loads the best solution
    path = list(os.path.split(sys.argv[1]))
    problem = path[-1].split('.')[0]
    path[-1] = 'solutions.txt'
    with open(os.path.join(*path)) as f:
        lines = f.read().splitlines()
        for line in lines:
            if problem in line:
                best_solution = line.split(' : ')[1]
        
    # build nodes
    nodes = [Node(vertex) for vertex in vertices]
    Indiv.nodes = nodes
    Indiv.n = len(vertices)
    
    # load config parameters
    config = yaml.load(file(sys.argv[0].replace('.py', '_config.yml')))
        
    best_of_the_best_costs = []
    gen = [range(1, Indiv.n) for i in xrange(config['pop_size'])]
    t = time()
    print 'Best known solution is', best_solution
    print 'Using config:\n', yaml.dump(config,default_flow_style=False)
    
    
    if config['use_mp']:
        # multi processing: build all processes
        q = Queue()
        processes = []    
        for i in xrange(config['tries']):
            # build starting configuration
            for ordering in gen:
                random.shuffle(ordering)
            # build corresponding process
            processes.append(Process(target=ga_min, args=([Indiv(ordering) for ordering in gen], config, q)))
        # start processes
        for p in processes:
            p.start()
        # get results
        for p in processes:
            p.join()
        # extract
        for i in xrange(config['tries']):
            best_indiv,best_costs = q.get()
            if i == 0:
                best_of_the_bests = best_indiv
            elif best_indiv.cost < best_of_the_bests.cost:
                best_of_the_bests = best_indiv
            best_of_the_best_costs.append(best_costs)
        
    else:
        for i in xrange(config['tries']):
            print 'Starting from a new population'
            # build first generation with shuffle ordering
            for ordering in gen:
                random.shuffle(ordering)
            population = [Indiv(ordering) for ordering in gen]

            # solve
            best_indiv,best_costs = ga_min(population, config)
            
            # compare to previous tries
            if i == 0:
                best_of_the_bests = best_indiv
            elif best_indiv.cost < best_of_the_bests.cost:
                best_of_the_bests = best_indiv
            best_of_the_best_costs.append(best_costs)
    print ' Found solution with cost:', best_of_the_bests.cost, 'vs best known', best_solution, 'in', round((time()-t), 2), 's'
        
    
    pylab.close('all')
    for costs in best_of_the_best_costs:
        pylab.plot(costs)
    pylab.xlabel('iterations')
    pylab.ylabel('cost')
    pylab.show()

