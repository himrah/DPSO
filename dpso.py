import networkx as nx
import numpy as np
from collections import Counter
import time
import operator
import matplotlib.pyplot as plt

class Particle:
	def __init__(self):
		self.pbest = []
		self.gbest = []
		self.gbest_mod = 0
		self.velocity = []
		self.G = nx.Graph()
		self.particle = []
		self.file_name = 'dolphin.txt'
		self.number_of_particles = 10
		self.modularity = 0
		self.iteration = 20

	def Input_Graph(self):
		temp=open(self.file_name,'r').read().split('\n')
		graph=[]
		for i in temp:
			t=[]
			for j in i.split():
				if(j):
					t.append(int(j))
			#print(t)
			if(t):
				graph.append(tuple(t))
		
		#print(graph)
		self.G.add_edges_from(graph)
		#print(self.G)
		j=0
		for i in self.G:
			#print(i)
			self.G.node[i]={'pos':j}
			j+=1

	def updatepos(self,graph):
		j=0
		for i in graph:
			n=[]
			if(self.velocity[j]):
				temp=graph.neighbors(i)
				for k in temp:
					n.append(graph.node[k]['pos'])
					#print(n)
				if(len(n)!=len(set(n))):
					p=Counter(n).most_common(1)[0][0]
					graph.node[i]['pos']=p
				else:
					if(graph.node[i]['pos'] in n):
						pass
					else:
						p=np.random.choice(n)
						graph.node[i]['pos']=p
			j+=1
		return graph.copy()
	

	def iweight(self,k):
		wmax=0.9
		wmin=0.4
		kmax=self.iteration
		w=((wmax-wmin)*((kmax-k)/kmax))+wmin
		return w


	def updatevelocity(self,graph,k):
		c1=c2=1.494
		w=self.iweight(k)
		v1=[]
		v2=[]
		v3=[]
		v4=[]
		v5=[]
		j=0
		for i in graph:
			v1.append(int((self.pbest[j]==graph.node[i]['pos']) and '0' or '1'))
			v2.append(int((self.gbest[j]==graph.node[i]['pos']) and '0' or '1'))
			r1=float(np.round(np.random.uniform(0.1,0.9),3))
			r2=float(np.round(np.random.uniform(0.1,0.9),3))
			R1=c1*r1
			R2=c2*r2
			v3.append(v1[j]*R1)
			v4.append(v2[j]*R2)
			v5.append(v3[j]+v4[j]+(self.velocity[j]*w))
			self.velocity[j]=(int((v5[j]>=1) and '1' or '0'))
			j+=1

	def particle_init(self):
		self.particle=[]
		#j=1
		a=self.G.nodes()
		l=np.random.randint(1,len(a),len(a)).tolist()
		self.pbest=l
		#initlization base on same neighbors
		for j in range(self.number_of_particles):
			copy=self.G.copy()
			num=np.random.randint(1,len(a))
			temp=copy.neighbors(num)
			for i in temp:
				copy.node[i]['pos']=num
			self.particle.append(copy)

	def fitness(self,graph):
		m=graph.number_of_edges()
		l=1/(2*m)
		temp=0
		for j in graph:	
			for i in graph:
				A=int(i in graph.neighbors(j))
				k1=len(graph.neighbors(j))
				k2=len(graph.neighbors(i))
				gama=int(graph.node[j]['pos']==graph.node[i]['pos'])
				temp+=((A-(k1*k2)/(2*m))*gama)
			
		mod=temp*l
                
		return np.round(mod,4)

	def rearrange(self,graph):
		pos=[]
		node=graph.nodes()
		for i in graph:
			pos.append(graph.node[i]['pos'])
		new_pos=[]
		single=list(set(pos))
		for i in single:
			if(i in node):
				node.remove(i)
				#print(node)
				f=True
			else:
				f=False	
			num=np.random.choice(node)
			new_pos.append(num)
			node.remove(num)
			if(f is True):
				node.append(i)
		for i in graph:
			t=graph.node[i]['pos']
			d=single.index(t)
			graph.node[i]['pos']=new_pos[d]
			#print(graph.node[i]['pos'])
		return graph.copy()			



	def gbest_init(self,particle):
		best=[]
		for i in range(particle[0].number_of_nodes()):
			best.append(0)
		f=-1
		for p in particle:
			t=self.fitness(p)
			if(t>f):
				j=0
				f=t
				for k in p:
					best[j]=p.node[k]['pos']
					j+=1
		self.gbest_mod=t
		self.gbest=best


	def optimize(self):
		startTime = time.time()
		self.Input_Graph()
		self.particle_init()
		#print(self.particle)
		self.gbest_init(self.particle)
		t=self.G.number_of_nodes()
		vel=[]
		for ll in range(t):
			vel.append(0)
		self.velocity=vel
		for i in range(self.iteration):
			print("Iteration : %d"%(i+1),end='\r')
			for p in self.particle:
				self.updatevelocity(p,i+1)
				t1=self.updatepos(p)
				t2=self.rearrange(t1)
				
				for r in p:
					p.node[r]['pos']=t2.node[r]['pos']
				m=self.fitness(t2)
				#print("best modularity : %f"%m,end='\r')
				if(m>self.modularity):
					self.modularity=m
					k=0
					for d in t2:
						self.pbest[k]=t2.node[d]['pos']
						k+=1

			if(self.modularity>self.gbest_mod):
				self.gbest_mod=self.modularity
				
				self.gbest=self.pbest
		
		j=0		
		for i in self.G:
			self.G.node[i]['pos']=self.gbest[j]
			j+=1
		print("\n\n**********************************************************")	
		print('\nThe script take {0} second '.format(np.round((time.time() - startTime),2)))
		print("\nModularity is : ",self.gbest_mod)
		print("\nNumber of Communites : ",len(set(self.gbest)))
		print("\nGlobal Best Position : ",self.gbest)
		print("\nGraph : ",self.G.node)
		nx.draw(self.G,node_color=[self.G.node[i]['pos'] for i in self.G])
		plt.show()

if __name__=='__main__':
	f=Particle()
	f.optimize()
