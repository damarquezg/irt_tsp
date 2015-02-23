#!/usr/bin/env python


from lxml import etree
import sys, os
import random
import yaml
from time import time
import pylab
from profilestats import profile
from heapq import nsmallest
from multiprocessing import Pool

pylab.ion()

def my_randint(start, length):
    return start + int(length*random.random())
                     
class Indiv:
    '''
    An individual is a given ordering 
    The cost of an individual is the cost to do a round trip
    '''
    def __init__(self, ordering):
        self.ordering = [0] + ordering
    def compute_cost(self):
        self.cost = sum([self.nodes[self.ordering[i-1]][self.ordering[i]] for i in xrange(self.n)])
    def cross_and_mutate(self, other):
        '''
        Performs a crossing between this and the other
        Mutates the new individual and returns it
        '''
        # random index
        #print 'self', self.ordering
        #print 'other', other.ordering
        #n = random.randint(2,self.n-2)
        n = my_randint(2, self.n-3)
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
        
        # mutation
        n1 = my_randint(0, self.n-1)
        n2 = my_randint(0, self.n-1)
        while n1 == n2:
            n2 = my_randint(0, self.n-1)
        # switch
        (ordering[n1],ordering[n2]) = (ordering[n2],ordering[n1])
        # build and return new indiv
        return Indiv(ordering)
   
        
 
def compare(start, end):
    n = my_randint(start, end-start)
    n = random.randint(start, end)

@profile() 
def ga_min(population, config):
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
   
    # compute cost of initial individuals
    for indiv in population:
        indiv.compute_cost()
    costs_and_idx = [(indiv.cost,i) for i,indiv in enumerate(population)]
    # extract best ones
    best_costs_and_idx = nsmallest(keep_best, costs_and_idx)    # all n best costs and indices                        
    best_cost = best_costs_and_idx[0][0]  # the actual best cost
    best_costs = []
    
    it = 0
    iter_follow = 0
    while it < max_iter and iter_follow < iter_out:
        it += 1
        
        # elitism: build new population from best individuals
        new_pop = [population[value[1]] for value in best_costs_and_idx]#costs.index(best_cost)] for best_cost in nbest]
            
        # selection
        # 1 vs 1 tournaments to fill up half the new population from the previous one
        #competitors = []
        for i in xrange(half_pop - keep_best):
            #n1 = random.randint(0,pop_size-1)
            n1 = my_randint(0, pop_size)
            #while n1 in competitors:
            #    n1 = random.randint(0,pop_size-1)
            #competitors.append(n1)
            n2 = my_randint(0, pop_size)
            #while n2 in competitors or n2 == n1:
            #    n2 = random.randint(0,pop_size-1)
            #competitors.append(n2)
            if population[n1].cost < population[n2].cost:
                new_pop.append(population[n1])
            else:
                new_pop.append(population[n2])  
        
        # reproduction and mutation
        for i in xrange(half_pop):
            n1 = my_randint(0, half_pop)
            n2 = my_randint(0, half_pop)
            new_pop.append(new_pop[n1].cross_and_mutate(new_pop[n2]))
            
        # compute cost of new individuals
        for i in xrange(half_pop, pop_size):
            new_pop[i].compute_cost()
           
        # update population
        population = new_pop
        # get new best individuals
        costs_and_idx = [(indiv.cost,i) for i,indiv in enumerate(population)]
        best_costs_and_idx = nsmallest(keep_best, costs_and_idx)
        best_idx = best_costs_and_idx[0][1]
        
        # check for best indiv
        if best_cost > population[best_idx].cost:
            # new best individual, save it 
            best_cost = population[best_idx].cost
            iter_follow = 0
        else:
            iter_follow += 1
        best_costs.append(best_cost)
        
    print '  Found best cost', best_cost, 'in', it, 'iterations and', round((time()-t), 2), 's'
    return population[best_idx], best_costs
  

if __name__ == '__main__':
   

    # check for input file
    if len(sys.argv) < 1:
        print 'Give a problem file'
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
        
    # build dictionnary of edges
    Indiv.n = len(vertices)
    nodes = []
    max_cost = 0
    for vertex in vertices:
        nodes.append({})
        for edge in vertex.findall('edge'):
            cost = int(float(edge.get('cost')))
            nodes[-1][int(edge.text)] = cost
            if cost > max_cost:
                max_cost = cost
    # fill up impossible trips with maximal cost
    for i in xrange(Indiv.n):
        for j in xrange(Indiv.n):
            if j != i and j not in nodes[i]:
                nodes[i][j] = Indiv.n*max_cost
                print 'Wrote impossible trip from', i, 'to', j 
    Indiv.nodes = nodes
    
    
    # load config parameters
    config = yaml.load(file(sys.argv[0].replace('.py', '_config.yml')))
        
    best_of_the_best_costs = []
    gen = [range(1, Indiv.n) for i in xrange(config['pop_size'])]
    t = time()
    print 'Best known solution is', best_solution
    print 'Using config:\n', yaml.dump(config,default_flow_style=False)
    
    
    if config['use_mp']:
        # multi processing: build all processes
        pool = Pool() 
        results = []
        for i in xrange(config['tries']):
            # build starting configuration
            for ordering in gen:
                random.shuffle(ordering)
            # add this solver to pool
            results.append(pool.apply_async(ga_min, args=([Indiv(ordering) for ordering in gen], config)))

        # extract results
        for i in xrange(config['tries']):
            best_indiv,best_costs = results[i].get()
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
    pylab.plot(pylab.xlim(), [best_solution]*2, 'k--', label='best solution')
    pylab.legend()
    pylab.xlabel('iterations')
    pylab.ylabel('cost')
    pylab.show()

