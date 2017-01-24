import networkx as nx
import numpy as np
from collections import Counter

class Particle:
	def __init__(self):
		self.pbest=[]
		self.gbest=[]
		self.gbest_mod=0
		self.velocity=[]
		#self.position=nx.Graph()
		self.G=nx.Graph()
		#self.temp_graph=nx.Graph()
		#self.G.add_nodes_from([1,2,3,4,5,6,7,8,9],pos=0)
		#self.G.add_edges_from([(1,2),(2,5),(5,4),(4,3),(3,1),(5,6),(6,9),(6,8),(6,7),(7,8),(8,9),(1,4),(2,3)])
		self.particle=[]
		self.modularity=0
		self.itration=100


	def Input_Graph(self):
		temp=open('graph.txt','r').read().split()
		graph=[]
		for i in temp:
			t=[]
			for j in i.split('-'):
				t.append(int(j))
			graph.append(tuple(t))
		
		self.G.add_edges_from(graph)
		for i in self.G:
			self.G.node[i]={'pos':0}

	def updatepos(self,graph):
		#self.pos=np.add(self.pos,self.velocity)
		j=0
		for i in graph:
			n=[]
			if(self.velocity[j]):
				temp=graph.neighbors(i)
				#[n.append(position.node[k]['pos']) for k in temp]
				for k in temp:
					n.append(graph.node[k]['pos'])
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
		#self.position=graph.copy()
		return graph.copy()
		#return position


	def updatevelocity(self,graph):
		#R1=1.4
		#R2=0.7
		c1=c2=1.494
		w=1
		v1=[]
		v2=[]
		v3=[]
		v4=[]
		v5=[]
		j=0
		for i in graph:
			v1.append(int((self.pbest[j]==graph.node[i]['pos']) and '0' or '1'))
			v2.append(int((self.gbest[j]==graph.node[i]['pos']) and '0' or '1'))
			r1=np.random.choice([0,1])
			r2=np.random.choice([0,1])
			R1=c1*r1
			R2=c2*r2
			v3.append(v1[j]*R1)
			v4.append(v2[j]*R2)
			v5.append(v3[j]+v4[j]+(self.velocity[j]*w))
			self.velocity[j]=(int((v5[j]>=1) and '1' or '0'))
			j+=1


	def particle_init(self):
		self.particle=[]
		#l=np.random.randint(1,5,self.G.number_of_nodes()).tolist()

		#for i in self.pbest:
		a=self.G.nodes()
		a.reverse()

		l=np.random.randint(self.G.nodes()[0],a[0]+1,self.G.number_of_nodes()).tolist()
		self.pbest=l

		for i in range(10):
			a=self.G.nodes()
			a.reverse()
			l=np.random.randint(self.G.nodes()[0],a[0]+1,self.G.number_of_nodes()).tolist()
			p=0
			for j in self.G:
				self.G.node[j]['pos']=l[p]
				p+=1
			t=self.G.copy()
			self.particle.append(t)
		return self.particle	



	def fitness(self,graph):
		m=graph.number_of_edges()
		l=1/(2*m)
		temp=0
		#first=graph.nodes()[0]
		for j in graph:	
			for i in graph:
				A=int(i in graph.neighbors(j))
				k1=len(graph.neighbors(j))
				k2=len(graph.neighbors(i))
				gama=int(graph.node[j]['pos']==graph.node[i]['pos'])
				temp+=((A-(k1*k2)/(2*m))*gama)
			
		mod=temp*l
                
		return np.round(mod,2)

	def rearrange(self,graph):
		pos=[]
		node=graph.nodes()
		for i in graph:
			pos.append(graph.node[i]['pos'])
		#print("nodes : ",node)
		#print("position : ",pos)	
		new_pos=[]
		single=list(set(pos))
		node_copy=node
		for i in single:
			#print("node before : ",node)
			if(i in node):
				node.remove(i)
				f=True
			else:
				f=False	
			#print("After",node)
			num=np.random.choice(node)
			#print("Random Number : ",num)
			new_pos.append(num)
			node.remove(num)
			if(f is True):
				node.append(i)
		for i in graph:
			t=graph.node[i]['pos']
			d=single.index(t)
			graph.node[i]['pos']=new_pos[d]

		return graph.copy()			



	def gbest_init(self,particle):
		best=[]
		for i in range(particle[0].number_of_nodes()):
			best.append(0)
		f=-1
		for p in particle:
			t=self.fitness(p)
			#print(t)
			if(t>f):
				j=0
				f=t
				for k in p:
					best[j]=p.node[k]['pos']
					j+=1
				#print(best)	
		#return best
		self.gbest_mod=t
		self.gbest=best


	def optimize(self):
		self.__init__()
		self.Input_Graph()
		self.particle_init()                     #create particle of graph
		self.gbest_init(self.particle)
		t=self.G.number_of_nodes()
		for i in range(t):
			self.velocity.append(0)


		for i in range(self.itration):
			for p in self.particle:
				self.updatevelocity(p)
				t1=self.updatepos(p)
				t2=self.rearrange(t1)
				#for r in 
				#self.itration[i].node[]
				for r in p:
					p.node[r]['pos']=t2.node[r]['pos']
					#p.node[r]['pos']=t2.node[r]['pos']
				m=self.fitness(t2)

				if(m>=self.modularity):
				#pbest=x
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
		print("**********************************************************")	
		print("Modularity is : ",self.gbest_mod)
		print("\nGlobal Best Position : ",self.gbest)
		print("\nGraph : ",self.G.node)