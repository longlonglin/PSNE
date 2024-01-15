import numpy as np
import networkx as nx
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from multiprocessing import Pool
from tqdm import tqdm
import time
import random
from Utils import alias_draw, alias_setup
from BaseModel import BaseModel
import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
import math
from scipy.sparse import identity
import gc
from numba import jit
from numpy import float16
from numpy import int32 
from numpy import float32
from concurrent.futures import ProcessPoolExecutor


class PSNE_model(BaseModel):
    
    @staticmethod
    def add_args(parser):
        
        # fmt: off
        parser.add_argument("--window-size", type=int32, default=10,
                            help="Window size of approximate matrix. Default is 10.")
        parser.add_argument("--num-round", type=int32, default=100,
                            help="Number of round in PSNE. Default is 100.")
        parser.add_argument("--worker", type=int32, default=10,
                            help="Number of parallel workers. Default is 10.")
        parser.add_argument("--emb-size", type=int32, default=128)
        parser.add_argument("--a_decay", type=int32, default=0.1)
        parser.add_argument("--mu", type=int32, default=0.1,
                            help="different datasets should use different mu.")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.window_size, args.num_round, args.worker, args.a_decay)

    def __init__(self, dimension=128, window_size=10, mu=1, num_round=30, worker=10, a_decay=0.1):
        super(PSNE_model, self).__init__()
        self.dimension = dimension
        self.window_size = window_size
        self.worker = worker
        self.num_round = num_round
        self.a_decay = a_decay
        self.mu = mu
  
    def forward(self, graph='datapath',dataset_name='dataname'):
       
        self.graph = graph
        if dataset_name=='ppi' or dataset_name=='wiki':
            self.G = nx.read_edgelist(self.graph, nodetype=int32, create_using=nx.DiGraph(),edgetype=int32, data=True)
            matrix0 = scipy.sparse.lil_matrix((self.G.number_of_nodes(), self.G.number_of_nodes()),dtype=int32)
            for e in self.G.edges():
                if e[0] != e[1]:
                    matrix0[e[0], e[1]] = 1
                    matrix0[e[1], e[0]] = 1
        else:
            self.G = nx.read_edgelist(self.graph, delimiter=',', nodetype=int, create_using=nx.DiGraph())
            matrix0 = scipy.sparse.lil_matrix((self.G.number_of_nodes(), self.G.number_of_nodes()),dtype=int32)
            for e in self.G.edges():
                if e[0] != e[1]:
                    matrix0[e[0]-1, e[1]-1] = 1
                    matrix0[e[1]-1, e[0]-1] = 1

       
        self.G = nx.from_numpy_array(matrix0)
        
        del matrix0
        gc.collect()
        
        node2id = dict([(node, vid) for vid, node in enumerate(self.G.nodes())]) 
        self.is_directed = nx.is_directed(self.G) 
        self.num_node = self.G.number_of_nodes()  
        self.num_edge = self.G.number_of_edges()  
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]  

        id2node = dict(zip(node2id.values(), node2id.keys()))  

        self.num_neigh = np.asarray([len(list(self.G.neighbors(id2node[i]))) for i in range(self.num_node)])  
        self.neighbors = [[node2id[v] for v in self.G.neighbors(id2node[i])] for i in range(self.num_node)] 
        s = time.time() 
        self.alias_nodes = {}  
        self.node_weight = {}  

        for i in range(self.num_node):  
            unnormalized_probs = [self.G[id2node[i]][nbr].get("weight", 1.0) for nbr in self.G.neighbors(id2node[i])]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[i] = alias_setup(normalized_probs)
            self.node_weight[i] = dict(
                zip([node2id[nbr] for nbr in self.G.neighbors(id2node[i])], unnormalized_probs, ))

        t = time.time()
        print("alias_nodes", t - s)
        print("number of sample edges ", self.num_round * self.num_edge * self.window_size)
        print("random walk start...")
        t0 = time.time()
        results = []
        pool = ProcessPoolExecutor(self.worker)
        for i in range(self.worker):
            results.append(
                pool.submit(self._random_walk_matrix, i,))
        pool.shutdown()
        
        phase1 = time.time() - t0
        print("random walk time", phase1)
        
        matrix = sp.csr_matrix((self.num_node, self.num_node),dtype=float16)
        I_identity = sp.identity(self.num_node,dtype=float16, format='csr')  
        A = sp.csr_matrix(nx.adjacency_matrix(self.G),dtype=float16)

        A_ = A+I_identity  

        degree = sp.diags(np.array(A.sum(axis=0))[0], format="csr",dtype=float16)  
        degree_inv = degree.power(-1)

        degree_ = sp.diags(np.array(A_.sum(axis=0))[0], format="csr",dtype=float16)  
        degree_inv2 = degree_.power(-1 / 2)

        t1 = time.time()
        for res in results:
            matrix += res.result()
        t2 = time.time()
        phase2 = time.time() - t1
        print("construct random walk matrix time", phase2)
        
        L_ = sp.csgraph.laplacian(matrix, normed=False, return_diag=False,dtype=float16)

        a = self.a_decay
        ar = []  
        for i in range(1, self.window_size + 1):  
            ar.append(a * pow(1 - a, i))  
        sum_ = sum(ar)

        dad= degree_inv2.dot(A_).dot(degree_inv2)
        
        del degree_inv2,A_
        gc.collect()
        
        SaPPR = (sum_ * (I_identity - degree_inv.dot(L_)) + a * I_identity)   
       
        epsilon = 1/(self.num_node)  
        SaPPR.data[SaPPR.data <= epsilon] = 0
        SaPPR.eliminate_zeros()
     
        M = dad.dot(SaPPR)   
       
        del SaPPR
        gc.collect()

        M =self.mu*self.num_node*M   
        M.data[M.data <= 1] = 1
        M.data = np.log(M.data,dtype=float32)
        M.eliminate_zeros()
           
        phase3 = time.time() - t2

        print("number of nzz", M.nnz)
        print("construct sturct-aware PPR matrix  time", time.time() - t2)
        print(type(M.data[0]))
        embeddings, phase4= self._get_embedding_rand(M)
       
        print('total time:',phase1+phase2+phase3+phase4)
        del M
        gc.collect()
        
        return embeddings
        

    
    def _get_embedding_rand(self, matrix,):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]  
        smat = sp.csc_matrix(matrix,dtype=float16)
        print("svd sparse", smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        phase4 =time.time() - t1
        print("time for randomized tSVD ", phase4)
        return U, phase4

    
    
    
    def _path_sampling(self, u, v, r):
        # sample a r-length path from edge(u, v) and return path end node
        k = np.random.randint(r) + 1
        zp, rand_u, rand_v = 2.0 / self.node_weight[u][v], k - 1, r - k
        for i in range(rand_u):
            new_u = self.neighbors[u][alias_draw(self.alias_nodes[u][0], self.alias_nodes[u][1])]
            zp += 2.0 / self.node_weight[u][new_u]
            u = new_u
        for j in range(rand_v):
            new_v = self.neighbors[v][alias_draw(self.alias_nodes[v][0], self.alias_nodes[v][1])]
            zp += 2.0 / self.node_weight[v][new_v]
            v = new_v
        return u, v, zp
    
    def _random_walk_matrix(self, pid):
        np.random.seed(pid)
        matrix = sp.lil_matrix((self.num_node, self.num_node),dtype=float32)
        arr, value_list = self.get_arr_value_list() 
        
        for i in tqdm(range(self.num_edge * self.num_round // self.worker),mininterval=1):
           
            u, v = self.edges[i % self.num_edge]
            if not self.is_directed and np.random.rand() > 0.5:
                v, u = u, v
            for k in range(1, self.window_size + 1):
                r = self.get_lengthofstep(arr, value_list)  
                u_, v_, zp = self._path_sampling(u, v, r)
                matrix[u_, v_] += 2 * r / self.num_round / self.window_size / zp
        return matrix
    
    def get_lengthofstep(self, arr, value_list):
        num_random = random.random()
        sum_of_value = 0
        for index, value in enumerate(arr):  
            if sum_of_value > num_random:
                return value_list[index - 1]
            else:
                sum_of_value = value + sum_of_value
        return value_list[-1]
    
    def get_arr_value_list(self):
        a = self.a_decay
        ar = []  
        for i in range(1, self.window_size + 1):  
            ar.append(a * pow(1 - a, i))  
        ar = np.array(ar)

        ar = ar / sum(ar)  
        ar = list(ar)
        value_list = []
        count = 1
        for i in ar:  
            value_list.append(count)
            count += 1
        return ar, value_list
