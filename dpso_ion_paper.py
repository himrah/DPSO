import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import normalized_mutual_info_score as NMI
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
		self.file_name = 'footb.txt'
		self.synthetic = 'footballLabel.txt'
		self.pred = {}
		self.number_of_particles = 20
		self.modularity = 0
		self.iteration = 50

	def Input_Graph(self):
		temp=open(self.file_name,'r').read().split('\n')
		graph=[]
		for i in temp:
			t=[]
			for j in i.split():
				if(j):
					t.append(int(j))
			if(t):
				graph.append(tuple(t))
		self.G.add_edges_from(graph)
		j=0
		for i in self.G:
			self.G.node[i]={'pos':j}
			j+=1

		#self.pred	
		temp=open(self.synthetic,'r').read().split('\n')	
		for i in temp:
			m=[]
			for j in i.split():
				if(j):
					m.append(int(j))
			if(m):
				self.pred.update({m[0]:m[1]})
					
	def updatepos_simple(self,graph): #update position based on most common neighbors
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


	def updatepos_ion(self,graph):
		inc=0
		copy=graph.copy()
		for i in graph:
			n=[]
			if(self.velocity[inc]):
				temp=copy.neighbors(i)
				for k in temp:
					n.append(copy.node[k]['pos']) # label of neighbor nodes
				ion=defaultdict(list)
				for j in temp:
					dg=graph.degree(j)
					num=n.count(copy.node[j]['pos'])      # number of label count
					#ion.append(num*dg)
					ion[num*dg].append(j)
					#ion.update({i:num*dg})
				#print(ion)
				val=max(ion)
				#print("val ",val)
				if(len(ion[val])>1):
					#print("if", ion[val])
					graph.node[i]['pos']=np.random.choice(ion[val])
				else:
					#print("else ",ion[val])
					graph.node[i]['pos']=ion[val][0]						
				#if((Counter(ion.values()).most_common(1)[0][1]))
				#graph.node[i]['pos']=max(ion)
			inc+=1
		#print(graph.node)	
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

	#based on max degree
	def particle_init(self):
		a = self.G.nodes()
		l = np.random.randint(1,len(a),len(a)).tolist()
		self.pbest = l
		dgre={}
		for i in self.G:
			dgre.update({i:self.G.degree(i)})
		sort_degree = sorted(dgre.items(), key=operator.itemgetter(1),reverse=True)
		for i in range(self.number_of_particles):
			copy=self.G.copy()
			#print(sort_degree)
			num=sort_degree[i][0]
			#print(num)
			first_node_neigbors=copy.neighbors(num)
			dg={j:copy.degree(j) for j in first_node_neigbors}
			#dg={j:g.degree(j) for j in n}
			t=sorted(dg.items(), key=operator.itemgetter(1),reverse=True)
			num1=t[0][0]
			second_node_neigbors=copy.neighbors(num1)
			node=list(set(first_node_neigbors).intersection(second_node_neigbors))
			node+=[num,num1]
			for j in node:
				copy.node[j]['pos']=num

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


	def modular_density(self,graph):
		community=defaultdict(list)
		for i in graph:
			community[graph.node[i]['pos']].append(i)	# community{label[node,node,node...],label[node,node,node...]}
		md=0
		l=0.3 # lambda
		for com in community:
			
			nd=[]
			for i in community[com]:
				temp=0
				nd.append(i)
				n=graph.neighbors(i)      
				for j in community[com]:  
					if(j in n):
						pass
					else:
						temp+=1       #////////// this is L(v1-v1')
						#print(temp)
			v1=(graph.subgraph(nd).number_of_edges())*2 #  this is just like L(v1,v1) function
			v2=(graph.subgraph(nd).number_of_nodes())
			md+=((2*l*v1)-(2*(1-l)*temp))/v2
			#print(md)
		#print(round(md,2))
		return round(md,2)	



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
			t=self.modular_density(p)
			if(t>f):
				j=0
				f=t
				for k in p:
					best[j]=p.node[k]['pos']
					j+=1
		self.gbest_mod=t
		self.gbest=best


	def optimize(self):
		#lll=[]
		it=0
		fit=[]
		tm=[]
		md=[]
		nmi=[]
		ittr=[]
		com=[]		
		for itr in range(10):
			self.__init__()
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
			#it=0
			print("Iteration : %d"%(itr+1),end='\r')
			for i in range(self.iteration):
				for p in self.particle:
					#print(p.node)
					self.updatevelocity(p,i+1)
					#t1=self.updatepos(p)
					#print(p.node)
					if(np.mod(i,2)):
						t1=self.updatepos_ion(p)
					else:
						t1=self.updatepos_simple(p)	
					#print(t1.node)
					t2=self.rearrange(t1)
					
					#for r in p:
					#	p.node[r]['pos']=t2.node[r]['pos']
					#n=NMI(list(self.pred.values()),[t2.node[nd]['pos'] for nd in t2])
					#print(n)
					m=self.modular_density(t2)
					
					#print("best modularity : %f"%m,end='\r')
					if(m>self.modularity):
						self.modularity=m
						k=0
						#it+=1
						for d in t2:
							self.pbest[k]=t2.node[d]['pos']
							k+=1

				if(self.modularity>self.gbest_mod):
					self.gbest_mod=self.modularity
					it=i+1
					#print(it)
					print(self.gbest_mod)
					self.gbest=self.pbest
			
			j=0		
			for i in self.G:
				self.G.node[i]['pos']=self.gbest[j]
				j+=1

			#fit=self.fitness(self.G)
			n=NMI(list(self.pred.values()),self.gbest)	




			n=NMI(list(self.pred.values()),self.gbest)
			fit.append(self.fitness(self.G))
			tm.append(float(format(np.round((time.time() - startTime),2))))
			md.append(self.gbest_mod)
			nmi.append(n)
			ittr.append(it)
			com.append(len(set(self.gbest)))



		print('\nThe script take {0} second ',np.average(tm))
		print("\nModular density is : ",np.average(md))
		print("\nModularity is : ",np.average(fit))
		print("\nNMI : ",np.average(nmi))
		print("\nItration : ",np.average(ittr))
		print("\nNumber of Communites : ",np.average(com))	

		print('\nThe script take {0} second ',max(tm))
		print("\nModular density is : ",max(md))
		print("\nModularity is : ",max(fit))
		print("\nNMI : ",max(nmi))
		print("\nItration : ",max(ittr))
		print("\nNumber of Communites : ",max(com))
		nx.draw(self.G,node_color=[self.G.node[i]['pos'] for i in self.G])
		plt.show()




		"""print("\n\n**********************************************************")	
		print('\nThe script take {0} second '.format(np.round((time.time() - startTime),2)))
		print("\nModular density is : ",self.gbest_mod)
		print("\nModularity is : ",fit)
		print("\nNMI : ",n)
		print("\nNumber of Communites : ",len(set(self.gbest)))
		print("\nGlobal Best Position : ",self.gbest)"""
		#for jjj in range(10)
		#lll[jjj]+=format(np.round((time.time() - startTime),2))
		#lll[]
		#print("\nGraph : ",self.G.node)
		#nx.draw(self.G,node_color=[self.G.node[i]['pos'] for i in self.G])
		#plt.show()

if __name__=='__main__':
	f=Particle()
	f.optimize()



