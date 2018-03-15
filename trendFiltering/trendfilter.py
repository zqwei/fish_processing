from scipy.sparse import dok_matrix
import random, pdb
import numpy as np
from scipy.sparse import linalg as splinalg
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time

def generate_diff(order = 1, size = None):
    'Recursively generate difference operator matrix'
    assert(type(size) is type(10))
    if order == 1:
        m = dok_matrix((size-1,size))
        for i in range(size-1):
            m[i,i] = 1
            m[i,i+1] = -1
        return m

    else:
        return generate_diff(order= 1, size= size - order + 1)*generate_diff(order = order -1,size = size)




class partition(object):
    default_part = {'pos': [], 'neg': [], 'act': []}
    def __init__(self,size,**argv):
        self.size = size
        self.part = deepcopy(partition.default_part)
        self.part.update(**argv)

    def randomize(self):
        self.part = deepcopy(partition.default_part)
        for i in range(self.size):
            s = random.choice([self.part['pos'],self.part['neg'],self.part['act']])
            s.append(i)
            
class TF(object):
    default_option = {'max_it':1000, 'opt_tol':1.0e-6}
    def __init__(self,y,lam,order=1,mode = -1):
        self.y = y
        self.lam = lam
        self.order = order
        self.size = max(y.shape)
        self.D = generate_diff(order = order,size=self.size)
        self.D = self.D.tocsr()
        self.P = partition(self.size - order)
        self.P.randomize()
        self.x = np.zeros((self.size,1))
        self.z = np.zeros((self.size-order,1))
        self.mode = mode
        
        self.info = {'status': 'Initialized'}
        self.info['iter'] = 0
        self.info['time'] = None
        self.silence = False
        self.collector = {'obj':[],'vio':[],'|vio|':[]}

    def new_partition(self,violation):
        'Obtain a new partition'
        for item in violation:
            vfrom = item['vfrom']
            vto = item['vto']
            self.P.part[vfrom] = np.setdiff1d(self.P.part[vfrom],item['what'],assume_unique = True)
            self.P.part[vfrom] = [int(i) for i in self.P.part[vfrom]]
            self.P.part[vto] = np.union1d(self.P.part[vto],item['what'])
            self.P.part[vto] = [int(i) for i in self.P.part[vto]]

    def new_solution(self):
        'Obtain a new primal-dual solution'
        P, N, A = (self.P.part['pos'],self.P.part['neg'],self.P.part['act'])
        I = P + N
        self.z[P] = 1
        self.z[N] = self.mode
        if len(A) > 0:
            Lhs = self.D[A,:]*self.D[A,:].T
            rhs = self.D[A,:]*self.y/self.lam - self.D[A,:]*self.D[I,:].T*self.z[I]
            self.z[A] = splinalg.cg(Lhs.tocsr(),rhs,self.z[A])[0][:,np.newaxis]
        self.x = self.y - self.lam*self.D.T*self.z

    def check_violation(self):
        'Evaluate violations'
        P, N, A = ('pos','neg','act')
        violation = []
        sP, sN, sA = (self.P.part['pos'],self.P.part['neg'],self.P.part['act'])
        Dx = self.D*self.x
        violation.append({'vfrom': P, 'vto': A, 'what': [i for i in sP if Dx[i] < 0]})
        violation.append({'vfrom': N, 'vto': A, 'what': [i for i in sN if Dx[i] > 0]})
        violation.append({'vfrom': A, 'vto': P, 'what': [i for i in sA if self.z[i] > 1]})
        violation.append({'vfrom': A, 'vto': N, 'what': [i for i in sA if self.z[i] < self.mode]})
        return violation

    @property
    def obj(self):
        'Calculate objective function value'
        return 0.5*np.linalg.norm(self.x - self.y) + self.lam*np.linalg.norm(self.D*self.x,1)
    @property
    def Dx(self):
        return self.D*self.x

    @property
    def title(self):
        #'{it:>4d} {obj:^6.2e} {vio1:^6.2e} {vio0:>4d}'
        t = '{0:>4s}'.format('IT')
        t+= '{0:>10s}'.format('OBJ')
        t+= '{0:>6s}'.format('#VIO')
        t+= '{0:>10s}'.format('|VIO|')
        t+= '{0:>6s}'.format('|neg|')
        t+= '{0:>6s}'.format('|act|')
        t+= '{0:>6s}'.format('|pos|')
        line = '-'*(len(t)+ 3)
        return line + '\n' + t + '\n' + line

    def cur_it(self,v):
        'Information of current iteration'
        cur = '{0:>4d}'.format(self.info['iter'])
        cur+= '{0:>10.2e}'.format(self.obj)
        pz = np.array([min(max(i[0],self.mode),1) for i in self.z])[:,np.newaxis]
        Dx = self.D*self.x
        mp = {'pos':Dx, 'neg':Dx, 'act':self.z - pz}
        vio = [j for i in v for j in mp[i['vfrom']][i['what']]]
        cur+= '{0:>6d}'.format(len(vio))
        cur+= '{0:>10.2e}'.format(np.linalg.norm(vio,1))
        cur+= '{0:>6d}'.format(len(self.P.part['neg']))
        cur+= '{0:>6d}'.format(len(self.P.part['act']))
        cur+= '{0:>6d}'.format(len(self.P.part['pos']))
        self.collector['obj'].append(self.obj)
        self.collector['vio'].append(len(vio))
        self.collector['|vio|'].append(np.linalg.norm(vio,1))
        return cur

    def pdas(self):
        'Apply PDAS to solve the problem'
        print(self.title)
        start = time()
        for i in range(TF.default_option['max_it']):
            self.new_solution()
            vio = self.check_violation()
            self.info['iter'] += 1
            if not self.silence: print(self.cur_it(vio))
            if sum([len(i['what']) for i in vio]) == 0:
                self.info['status'] = 'optimal'
                self.info['time'] = time() - start
                return

            self.new_partition(vio)

    def plot(self,name='collector.pdf'):
        'Plot trend'
        collector = self.collector
        plt.figure(figsize = (5,8))
        plt.title('Trend of indicators')
        plt.subplot(4,1,1)
        plt.plot(collector['obj'],'ro-')
        plt.xlabel('Iteration')
        plt.ylabel('Objective')

        plt.subplot(4,1,2)
        plt.plot(collector['vio'],'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('# violations')

        plt.subplot(4,1,3)
        plt.plot(collector['|vio|'],'go-')
        plt.xlabel('Iteration')
        plt.ylabel('|violations|')

        plt.subplot(4,1,4)
        plt.plot(self.y)
        plt.plot(self.x,'r')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.savefig(name)

class TFsafe(TF):
    'PDAS for TF with safe-guard'
    def __init__(self,y,lam,order=1,mode = -1,maxv=5):
        self.maxv = maxv
        super(TFsafe,self).__init__(y,lam,order=order,mode = mode,)
        self.vc = self.size + 1
        self.t = 0

    def max_vio(self,vio):
        'Find the maximum violation'
        P, N, A = ('pos','neg','act')
        violation = []
        sP, sN, sA = (self.P.part['pos'],self.P.part['neg'],self.P.part['act'])
        Dx = self.D*self.x
        try:
            violation.append({'vfrom': P, 'vto': A, 'what': max([(i,-Dx[i]) for i in sP if Dx[i] < 0],key = lambda s: s[1])})
        except ValueError:
            violation.append({'vfrom': P, 'vto': A, 'what':(-1,-1)})

        try:
            violation.append({'vfrom': N, 'vto': A, 'what': max([(i,Dx[i]) for i in sN if Dx[i] > 0],key = lambda s: s[1])})
        except ValueError:
            violation.append({'vfrom': N, 'vto': A, 'what':(-1,-1)})

        try:
            violation.append({'vfrom': A, 'vto': P, 'what': max([(i,self.z[i]-1) for i in sA if self.z[i] > 1],key = lambda s: s[1])})
        except ValueError:
            violation.append({'vfrom': A, 'vto': P, 'what':(-1,-1)})

        try:
            violation.append({'vfrom': A, 'vto': N, 'what': max([(i,self.mode-self.z[i]) for i in sA if self.z[i] < self.mode],key = lambda s: s[1])})
        except ValueError:
            violation.append({'vfrom': A, 'vto': N, 'what':(-1,-1)})

        violation = max(violation,key = lambda s: s['what'][1])
        violation['what'] = [violation['what'][0]]
        return [violation]

    def max_ind(self,vio):
        'Find the max index of violation'
        P, N, A = ('pos','neg','act')
        violation = []
        sP, sN, sA = (self.P.part['pos'],self.P.part['neg'],self.P.part['act'])
        Dx = self.D*self.x
        try:
            violation.append({'vfrom': P, 'vto': A, 'what': max([i for i in sP if Dx[i] < 0])})
        except ValueError:
            violation.append({'vfrom': P, 'vto': A, 'what': -1})
            
        try:
            violation.append({'vfrom': N, 'vto': A, 'what': max([i for i in sN if Dx[i] > 0])})
        except ValueError:
            violation.append({'vfrom': N, 'vto': A, 'what': -1})

        try:
            violation.append({'vfrom': A, 'vto': P, 'what': max([i for i in sA if self.z[i] > 1])})

        except ValueError:
            violation.append({'vfrom': A, 'vto': P, 'what': -1})

        try:
            violation.append({'vfrom': A, 'vto': N, 'what': max([i for i in sA if self.z[i] < self.mode])})
        except ValueError:
            violation.append({'vfrom': A, 'vto': N, 'what': -1})

        violation = max(violation,key = lambda s: s['what'])
        violation['what'] = [violation['what']]
        return [violation]

    def pdas(self):
        'Apply PDAS to solve the problem'
        print(self.title)
        start = time()
        for i in range(TF.default_option['max_it']):
            self.new_solution()
            vio = self.check_violation()
            self.info['iter'] += 1
            if not self.silence: print(self.cur_it(vio))
            vn = sum([len(i['what']) for i in vio])
            if vn == 0:
                self.info['status'] = 'optimal'
                self.info['time'] = time() - start
                return

            if vn < self.vc: 
                self.t = 0
                self.vc = vn
            else: 
                self.t += 1

            if self.t < self.maxv:
                self.new_partition(vio)
            else:
                print('safeguard invoked')
                self.new_partition(self.max_ind(vio))

    def pdas2(self):
        'Another safeguard strategy'
        print(self.title)
        R = rolling()
        start = time()
        for i in range(TF.default_option['max_it']):
            self.new_solution()
            vio = self.check_violation()
            self.info['iter'] += 1
            if not self.silence: print(self.cur_it(vio))
            vn = sum([len(i['what']) for i in vio])
            # Collect number of violations
            R.append(vn)
            if vn == 0:
                self.info['status'] = 'optimal'
                self.info['time'] = time() - start
                return

            if vn < self.vc: 
                self.t = 0
            else: 
                self.t += 1

            self.vc = R.mean

            if self.t < self.maxv:
                self.new_partition(vio)
            else:
                print('safeguard invoked')
                self.new_partition(self.max_ind(vio))


class rolling(object):
    'A help class to store rolling average'
    def __init__(self,size = 5):
        self.size = size
        self.array = []
        self.mean = 0
        self.cur = 0

    def append(self,item):
        'Append one item'
        if len(self.array) < self.size:
            self.array.append(item)
            self.mean = float(sum(self.array))/len(self.array)
        else:
            self.mean -= self.array[self.cur]/self.size
            self.array[self.cur] = item
            self.mean += item/self.size            
            self.cur = (self.cur + 1)%self.size

class TFsafeG(TFsafe):
    'PDAS for TF with safe-guard, greedy roll back to best available'
    def __init__(self,y,lam,order=1,mode = -1,maxv=5):
        super(TFsafeG,self).__init__(y,lam,order=order,mode = mode,maxv = maxv)
        self.bestP = deepcopy(self.P)
        self.bestkkt = np.inf

    def _roll_back(self):
        'Roll back to best available partition'
        self.P = deepcopy(self.bestP)

    def kkt(self,v):
        Dx = self.Dx
        pz = np.array([min(max(i[0],self.mode),1) for i in self.z])[:,np.newaxis]
        mp = {'pos':Dx, 'neg':Dx, 'act':self.z - pz}
        vio = [j for i in v for j in mp[i['vfrom']][i['what']]]
        return np.linalg.norm(vio,1)

    def pdas(self):
        'Apply PDAS to solve the problem'
        print(self.title)
        start = time()
        for i in range(TF.default_option['max_it']):
            self.new_solution()
            vio = self.check_violation()
            self.info['iter'] += 1
            if not self.silence: print(self.cur_it(vio))
            vn = sum([len(i['what']) for i in vio])
            if vn == 0:
                self.info['status'] = 'optimal'
                self.info['time'] = time() - start
                return

            if vn < self.vc:
                # A newer best available partition found
                self.t = 0
                self.vc = vn
                self.bestP = deepcopy(self.P)
            else: 
                # Violation up by once
                self.t += 1

            if self.t < self.maxv:
                self.new_partition(vio)
            else:
                print('safeguard invoked')
                self._roll_back()
                vio = self.check_violation()
                vn = sum([len(i['what']) for i in vio])
                while vn >= self.vc:
                    self.new_partition(self.max_ind(vio))
                    self.new_solution()
                    vio = self.check_violation()
                    self.info['iter'] += 1
                    print(self.cur_it(vio))
                    if not self.silence: vn = sum([len(i['what']) for i in vio])

                    if vn == 0:
                        self.info['status'] = 'optimal'
                        self.info['time'] = time() - start
                        return

    def pdas2(self):
        'Apply PDAS to solve the problem'
        print(self.title)
        start = time()
        for i in range(TF.default_option['max_it']):
            self.new_solution()
            vio = self.check_violation()
            self.info['iter'] += 1
            if not self.silence: print(self.cur_it(vio))
            vn = self.kkt(vio)
            if vn < 1.0e-6:
                self.info['status'] = 'optimal'
                self.info['time'] = time() - start
                return

            if vn < self.bestkkt:
                # A newer best available partition found
                self.t = 0
                self.bestkkt = vn
                self.bestP = deepcopy(self.P)
            else: 
                # Violation up by once
                self.t += 1

            if self.t < self.maxv:
                self.new_partition(vio)
            else:
                print('safeguard invoked')
                self._roll_back()
                vio = self.check_violation()
                vn = self.kkt(vio)
                while vn >= self.bestkkt:
                    self.new_partition(self.max_ind(vio))
                    self.new_solution()
                    vio = self.check_violation()
                    self.info['iter'] += 1
                    if not self.silence: print(self.cur_it(vio))
                    vn = self.kkt(vio)
                    if vn < 1.0e-6:
                        self.info['status'] = 'optimal'
                        self.info['time'] = time() - start
                        return

if __name__=='__main__':
    t = rolling()
    for i in range(1,20):
        t.append(i)
        print(t.cur,t.array,t.mean)
    D = generate_diff(order = 2, size = 4)
    D2 = generate_diff(order = 1, size = 4)
    print(D.todense())
    print(D2.todense())

    d = TF(np.random.randn(2000,1),0.5,order=2,mode = 0)
    d.pdas()
    d.plot()
