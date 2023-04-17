import numpy as np
from files import *
import random
import copy
import sys
from CRJ_Genome import *
import pickle
class RC_CRJ:
    def __init__(self,Is_Adapt):
        # 种群
        self.population_size=40
        self.population=[]
        # 物种
        self.species=[]

        #进化概率
        # self.

        self.init_reservoir_num=15
        self.max_reservoir_num = 20

        self.tournament_size=3
        self.p_complexity=0.5

        self.Is_Adapt=Is_Adapt



    def initial_population(self):
        self.population.clear()
        for i in range(0,self.population_size):
            gene=CRJ_Genome(self.init_reservoir_num)
            gene.cal_fitness(sys.argv[3]+"/Train/",1)
            print("第",i,"个个体的fitness",gene.fitness)
            self.population.append(gene)


    def evolve(self,K):
        self.initial_population()
        with open('output/fitness_record','w+') as f:
            with open('output/test_fitness_record','w+') as ft:
                for k in range(0,K):
                    self.evolve_no_specia()
                    print("第",k,"次迭代最优：",self.population[0].fitness)
                    f.writelines(str(self.population[0].fitness))
                    f.writelines('\n')
                    np.save("output/w1",self.population[0].w1)
                    np.save("output/w2", self.population[0].w2)
                    np.save("output/w_out", self.population[0].w_out)
                    np.save("output/w_bool", self.population[0].w_bool)
                    f.flush()
                    self.population[0].change_to_cir("output/")

                    output = open(sys.argv[3] + "/" + "best_genome.pkl", 'wb')
                    g_str = pickle.dumps(self.population[0])
                    output.write(g_str)
                    output.close()

            #         每20次测试一次
                    if k%20==0:
                        with open(sys.argv[3]+"/"+"best_genome.pkl", 'rb') as file:
                            genome = pickle.loads(file.read())
                            genome.cal_fitness(sys.argv[3]+"/Test/",0)
                            print("第",k/20+1,"次测试fitness:",genome.fitness)
                            ft.writelines(str(genome.fitness))
                            ft.writelines('\n')
                            ft.flush()
            ft.close()
        f.close()



        self.population[0].change_to_cir()



    def evolve_no_specia(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        mutate_rate_range=[0.2,0.8]

        for p in range(1,self.population_size):
            mutate_rate=mutate_rate_range[0]+(mutate_rate_range[1]-mutate_rate_range[0])*p/self.population_size

            if random.random()>mutate_rate:
                parent1,parent2=self.__tournament_selection()
                temp_pop = copy.deepcopy(parent1)
                if parent1.reservoir_num!=parent2.reservoir_num:
                    if random.random()<self.p_complexity:
                        if parent1.reservoir_num>parent2.reservoir_num:
                            temp_pop=copy.deepcopy(parent1)
                        else:
                            temp_pop=copy.deepcopy(parent2)
                    else:
                        if parent1.fitness>parent2.fitness:
                            temp_pop=copy.deepcopy(parent1)
                        else:
                            temp_pop=copy.deepcopy(parent2)
                for i in range(0,temp_pop.reservoir_num):
                    if i < min(parent1.reservoir_num, parent2.reservoir_num):
                        for j in range(0,temp_pop.reservoir_num):
                            if j<min(parent1.reservoir_num, parent2.reservoir_num):
                                #储存池
                                if parent1.w_bool[i][j]!=0 and parent2.w_bool[i][j]!=0 and random.random()<0.5:
                                    temp_pop.w1[i][j]=(parent1.w1[i][j]+parent2.w1[i][j])/2
                                    temp_pop.w2[i][j] = (parent1.w2[i][j] + parent2.w2[i][j]) / 2
                                elif parent1.w_bool[i][j]!=0 and parent2.w_bool[i][j]!=0 and random.random()>=0.5 and random.random()<0.75:
                                    temp_pop.w1[i][j] = parent1.w1[i][j]
                                    temp_pop.w2[i][j] = parent1.w2[i][j]
                                elif parent1.w_bool[i][j]!=0 and parent2.w_bool[i][j]!=0 and random.random()>=0.75 and random.random()<1.0:
                                    temp_pop.w1[i][j] = parent2.w1[i][j]
                                    temp_pop.w2[i][j] = parent2.w2[i][j]
                                elif parent1.w_bool[i][j]==0 and parent2.w_bool[i][j]!=0:
                                    temp_pop.w1[i][j] = parent2.w1[i][j]
                                    temp_pop.w2[i][j] = parent2.w2[i][j]
                                else:
                                    temp_pop.w1[i][j] = parent1.w1[i][j]
                                    temp_pop.w2[i][j] = parent1.w2[i][j]


                if random.random()<mutate_rate:
                    # if self.Is_Adapt=="1":
                    #     temp_pop.rewire_ra()
                    #            变异权重
                    temp_pop.mutate_weight()
                    #             增加节点
                    if random.random() < 0.8 and temp_pop.reservoir_num < self.max_reservoir_num:
                        temp_pop.add_node()
                    #             变异步长
                    if random.random() < 0.8:
                        temp_pop.mutate_step()
                    #     变异输入连接
                    if random.random() < 0.8:
                        temp_pop.mutate_in_gnd()
                self.population[p] = copy.deepcopy(temp_pop)
            else:
                if self.Is_Adapt == "1":
                    self.population[p].rewire_ra()
                #            变异权重
                self.population[p].mutate_weight()
                #             增加节点
                if random.random() < 0.8 and self.population[p].reservoir_num < self.max_reservoir_num:
                    self.population[p].add_node()
                #             变异步长
                if random.random() < 0.8:
                    self.population[p].mutate_step()
                 #     变异输入连接
                if random.random() < 0.8:
                    self.population[p].mutate_in_gnd()



        #         计算fitness
        for p in range(0,self.population_size):
            self.population[p].cal_fitness(sys.argv[3]+"/Train/",1)
            print("第",p,"个个体的fitness",self.population[p].fitness)


    def __tournament_selection(self):
        temp_tournament: list = []

        while len(temp_tournament) is not self.tournament_size:
            rand_index = random.randrange(0, self.population_size)

            temp: CRJ_Genome = self.population[rand_index]
            if temp not in temp_tournament:
                temp_tournament.append(temp)


        temp_tournament = sorted(temp_tournament, key=lambda x: x.fitness, reverse=True)

        return temp_tournament[0], temp_tournament[1]



