import torch

class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        import math
        if not math.isnan(other):
            self.n += 1
            self.sum += other
            self.sumsq += other*other
        return self

    def __isub__(self, other):
        import math
        if not math.isnan(other):
            self.n += 1
            self.sum -= other
            self.sumsq += other*other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

    def var(self):
        from math import sqrt
        return sqrt(self.sumsq / max(self.n, 1) - self.mean()**2)

    def semean(self):
        from math import sqrt
        return self.var() / sqrt(max(self.n, 1))

class EasyPoissonBootstrapAcc:
    def __init__(self, batch_size, confidence=0.95, seed=2112):
        from math import ceil
        from numpy.random import default_rng
        
        self.n = 0
        self.batch_size = batch_size
        self.confidence = confidence
        self.samples = [ EasyAcc() for _ in range(int(ceil(3 / (1 - self.confidence)))) ]
        self.rng = default_rng(seed)
        
    def __iadd__(self, other):
        self.n += 1
        
        poissons = self.rng.poisson(lam=self.batch_size, size=len(self.samples)) / self.batch_size
        
        for n, (chirp, acc) in enumerate(zip(poissons, self.samples)):
            acc += (chirp if n > 0 else 1) * other
            
        return self
         
    def __isub__(self, other):
        return self.__iadd__(-other)
    
    def ci(self):
        import numpy
        quantiles = numpy.quantile(a=[ x.mean() for x in self.samples ],
                                   q=[1 - self.confidence, 0.5, self.confidence])
        return list(quantiles)
    
    def formatci(self):
        z = self.ci()
        return '[{:<.4f},{:<.4f}]'.format(z[0], z[2])
    
class Schema(object):
    def __init__(self, *, attributes, target, skipcol, data):
        super().__init__()
        
        schema = {}
        n = 0
        for kraw, v in attributes:
            k = kraw.lower()

            if k in skipcol:
                continue
                
            if isinstance(v, str):
                if v in ['INTEGER', 'REAL']:
                    if any(thisv is None for row in data for thisk, thisv in zip(attributes, row) if thisk[0].lower() == k):
                        assert k != target, (k, target)
                        schema[k] = (lambda i: (lambda z: (i+1, 1) if z is None else (i, z)))(n)
                        n += 2
                    else:
                        schema[k] = (lambda i: (lambda z: (i, z)))(n)
                        n += 1
                elif k == 'date':
                    import ciso8601
                    import time
                    
                    schema[k] = (lambda i: (lambda z: (i, time.mktime(ciso8601.parse_datetime(z).timetuple()))))(n)
                    n += 1
                elif v == 'STRING':
                    uniques = set([ thisv for row in data for thisk, thisv in zip(attributes, row) if thisk[0].lower() == k ])
                    schema[k] =  (lambda h: (lambda z: (h[z], 1)))({ z: (n + m) for m, z in enumerate(uniques) })
                    n += len(uniques)
                else:
                    assert False, (k, v)
            elif isinstance(v, list) and all((isinstance(z, str) for z in v)):
                assert k != target, (k, target)
                schema[k] = (lambda h: (lambda z: (h[z], 1)))({ z: (n + m) for m, z in enumerate(v) })
                n += len(v)
            else:
                assert False
                
            if k == target:
                n -= 1
                
        assert target in schema, (target, attributes)
                
        self.schema = schema
        self.target = target
        self.nfeatures = n 
        
    def featurize(self, colname, val):
        if colname in self.schema:
            yield self.schema[colname](val)

def makeData(filename, *, target, skipcol, skiprow):
    import arff
    import numpy
    
    data = arff.load(open(filename, 'r'))
    schema = Schema(attributes=data['attributes'], target=target, skipcol=skipcol, data=data['data'])
    
    Y = []
    X = []
    
    for row in data['data']:
        hashrow = { kraw[0].lower(): v for kraw, v in zip(data['attributes'], row) }
        
        if skiprow(hashrow):
            continue
        
        y = None
        x = [0]*schema.nfeatures
        for col, val in hashrow.items():
            if col == target:
                y = next(schema.featurize(col, val))[1]
            else:
                for f, vf in schema.featurize(col, val):
                    from numbers import Number
                    assert isinstance(vf, Number), (col, val, f, vf)
                    x[f] = vf
                    
        Y.append(y)
        X.append(x)

    Y = numpy.array(Y)
    Ymin, Ymax = numpy.min(Y), numpy.max(Y)
    Y = (Y - Ymin) / (Ymax - Ymin)
    X = numpy.array(X)
    Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(X, axis=0, keepdims=True)
    if numpy.any(Xmin >= Xmax):
        X = X[:,Xmin[0,:] < Xmax[0,:]]
        Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(X, axis=0, keepdims=True)
    assert numpy.all(Xmax > Xmin), [ (col, lb, ub) for col, (lb, ub) in enumerate(zip(Xmin[0,:], Xmax[0,:])) if lb >= ub ]
    X = (X - Xmin) / (Xmax - Xmin)
    
    return X, Y

class ArffToPytorch(torch.utils.data.Dataset):
    def __init__(self, filename, *, target, skipcol, skiprow):
        X, Y = makeData(filename, target=target, skipcol=skipcol, skiprow=skiprow)
        self.Xs = torch.Tensor(X)
        self.Ys = torch.Tensor(Y).unsqueeze(1)
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        # Select sample
        return self.Xs[index], self.Ys[index]

class CauchyTruncatedNormal(torch.nn.Module):
    def __init__(self, dobs, numrff, sigma, device, approxargmax):
        from math import pi, sqrt
        
        super().__init__()
        
        self.rffW = torch.nn.Parameter(torch.empty(dobs, numrff).cauchy_(sigma = sigma).to(device), 
                                       requires_grad=False)
        self.rffb = torch.nn.Parameter((2 * pi * torch.rand(numrff)).to(device),
                                       requires_grad=False)
        self.sqrtrff = torch.nn.Parameter(torch.Tensor([sqrt(numrff)]).to(device), 
                                          requires_grad=False)
        self.linear = torch.nn.Linear(in_features=numrff, out_features=2, device=device)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
        self.approxargmax = approxargmax
        
    def distparams(self, Xs):
        with torch.no_grad():
            rff = (torch.matmul(Xs, self.rffW) + self.rffb).cos() / self.sqrtrff
            
        pre = self.linear(rff)
        return self.sigmoid(pre[:,0:1]), 0.01 + self.softplus(pre[:,1:2])
    
    def forward(self, Xs, As):
        mu, sigma = self.distparams(Xs)
        dens = torch.erf((1 - mu) / sigma) + torch.erf(mu / sigma)
        nums = torch.erf((1 - mu) / sigma) + torch.erf((mu - As) / sigma)
        return As * nums / dens
    
    def argmaxone(self, mu, sigma):
        from math import erf
        import scipy.optimize as so
        
        const = erf((1 - mu) / sigma)
        res = so.minimize_scalar(fun=lambda z: -z * (const + erf((mu - z) / sigma)),
                                 bounds=[0, 1],
                                 method='bounded')
        assert res.success, res
        return res.x

    def argmax(self, Xs, resolution):
        with torch.no_grad():
            if self.approxargmax:
                from math import ceil
                nsamples = int(ceil(1/resolution))
                As = torch.rand(size=(Xs.shape[0], nsamples), device=Xs.device)
                weirdrv = torch.max(input=self.forward(Xs, As), dim=1, keepdim=True)
                fhatstar = weirdrv.values
                ahatstar = torch.gather(input=As, dim=1, index=weirdrv.indices)
                return fhatstar, ahatstar
            else:
                mu, sigma = self.distparams(Xs)
                ahatstar = torch.Tensor([ [ self.argmaxone(m, s) ] for m, s in zip(mu, sigma) ])
                return self.forward(Xs, ahatstar), ahatstar

class CorralFastIGW(object):
    def __init__(self, *, eta, gammamin, gammamax, nalgos, device):
        import numpy
        
        super().__init__()
        
        self.eta = eta / nalgos
        self.gammas = torch.Tensor(numpy.geomspace(gammamin, gammamax, nalgos)).to(device)
        self.invpalgo = torch.Tensor([ self.gammas.shape[0] ] * self.gammas.shape[0]).to(device)
        
    def update(self, algo, invprop, reward):
        import numpy
        from scipy import optimize
        
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward
        
        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)
        newinvpalgo = torch.scatter(input=self.invpalgo,
                                    dim=0,
                                    index=algo,
                                    src=weightedlosses,
                                    reduce='add')
                                    
        # just do this calc on the cpu
        invp = newinvpalgo.cpu().numpy() 
        invp += 1 - numpy.min(invp)
        Zlb = 0
        Zub = 1
        while (numpy.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2 
        root, res = optimize.brentq(lambda z: 1 - numpy.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res
        
        self.invpalgo = torch.Tensor(invp + root, device=self.invpalgo.device)

    def sample(self, fhat, X):
        N, _ = X.shape

        algosampler = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False)
        algo = algosampler.sample((N,))
        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1),
                                dim=1,
                                index=algo.unsqueeze(1))
        gamma = torch.gather(input=self.gammas.unsqueeze(0).expand(N, -1),
                             dim=1,
                             index=algo.unsqueeze(1))
        resolution = 1 / torch.max(gamma).item()
        fhatstar, ahatstar = fhat.argmax(X, resolution)
        
        rando = torch.rand(size=(N, 1), device=X.device)
        fhatrando = fhat(X, rando)
        probs = 1 / (1 + gamma * (1 - torch.clip(fhatrando / fhatstar, max=1)))
        unif = torch.rand(size=(N, 1), device=X.device)
        shouldexplore = (unif <= probs).long()
        return (ahatstar + shouldexplore * (rando - ahatstar)), algo, invpalgo, shouldexplore, ahatstar

def bestconstant(dataset):
    import numpy
        
    constreward = [EasyAcc() for _ in range(100)]
    pnosale = [EasyAcc() for _ in range(100)]
    generator = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for _, Ys in generator:
        with torch.no_grad():
            for n in range(len(constreward)):
                z = n / len(constreward)
                constreward[n] += torch.mean((Ys >= z).long() * z).item()
                pnosale[n] += torch.mean((Ys < z).float()).item()
    
    return { 'best_constant_reward': max(( (v.mean(), pnosale[n].mean(), n/len(constreward)) for n, v in enumerate(constreward) ))
           }

def bootstrap(data, stat, conf):
    from math import ceil
    import numpy as np
    
    samples = ceil(3 / conf)
    stats = []
    for _ in range(samples):
        stats.append(stat(np.random.choice(data, len(data))))
        
    return np.quantile(stats, q=[conf, 1-conf])
            
def compute_evar_ci(rewards, q):
    import numpy as np
    
    def f(m, data):
        return q * np.sum(np.square(np.clip(data - m, a_min=0, a_max=None))) + (1 - q) * np.sum(np.square(np.clip(m - data, a_min=0, a_max=None)))  
            
    def evar(data):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda m: f(m, data), bounds=(np.min(data), np.max(data)), method='bounded')
        assert res.success
        return res.x
    
    return bootstrap(rewards, evar, conf=0.05)
        
def learnOnline(dataset, *, q, seed, batch_size, modelfactory, initlr, tzero, eta, gammamin, gammamax, nalgos):
    import numpy as np
    import time
    
    torch.manual_seed(seed)
        
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = None
    l1_loss = torch.nn.L1Loss(reduction='none')
    log_loss = torch.nn.BCELoss(reduction='none')
    
    print('{:<5s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<10s}'.format(
            'n', 'loss', 'since last', 'rahat', 'since last', f'evarq({q})', 'since last', 'punder', 'since last', 'accept', 'since last', 'dt (sec)'),
          flush=True)
    avloss, sincelast, acc, accsincelast, avreward, rewardsincelast = [ 
        EasyPoissonBootstrapAcc(batch_size=batch_size) for _ in range(6) ]
    accept, acceptsincelast, punder, pundersincelast = [
        EasyPoissonBootstrapAcc(batch_size=batch_size) for _ in range(4) ]
    allrewards, allrewardssincelast = [], []
    
    for bno, (Xs, ys) in enumerate(generator):
        if model is None:
            from math import sqrt
            model = modelfactory(Xs)
            opt = torch.optim.Adam(( p for p in model.parameters() if p.requires_grad ), lr=initlr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda t: sqrt(tzero) / sqrt(tzero + t))
            sampler = CorralFastIGW(eta=eta, gammamin=gammamin, gammamax=gammamax, nalgos=nalgos, device=Xs.device)
            start = time.time()
            
        opt.zero_grad()
        
        with torch.no_grad():
            sample, algo, invpalgo, shouldexplore, ahatstar = sampler.sample(model, Xs)
            reward = (sample <= ys).long() * sample
            allrewards.append(reward.numpy())
            allrewardssincelast.append(reward.numpy())
            rahat = (ahatstar <= ys).long() * ahatstar
        
        score = model(Xs, sample)
        with torch.no_grad():
            factor = q * (score < reward).long() + (1 - q) * (score >= reward).long()
        loss = 2 * torch.mean(factor * log_loss(score, reward))
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            acc += torch.mean(rahat).item()
            accsincelast += torch.mean(rahat).item()
            avreward += torch.mean(reward).item()
            rewardsincelast += torch.mean(reward).item()
            punder += torch.mean((sample < ys).float()).item()
            pundersincelast += torch.mean((sample < ys).float()).item()
            avloss += loss.item()
            sincelast += loss.item()
            accept += torch.mean(shouldexplore.float()).item()
            acceptsincelast += torch.mean(shouldexplore.float()).item()
            sampler.update(algo, invpalgo, reward)

        if bno & (bno - 1) == 0:
            evar = compute_evar_ci(np.concatenate(allrewards, axis=None), q)
            evarsincelast = compute_evar_ci(np.concatenate(allrewardssincelast, axis=None), q)
            
            print('{:<5d}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<10.5f}'.format(
                    avloss.n, avloss.formatci(), sincelast.formatci(), acc.formatci(),
                    accsincelast.formatci(), f'[{evar[0]:.4f},{evar[1]:.4f}]', f'[{evarsincelast[0]:.4f},{evarsincelast[1]:.4f}]',
                    punder.formatci(), pundersincelast.formatci(),
                    accept.formatci(), acceptsincelast.formatci(),
                    time.time() - start),
                  flush=True)
            sincelast, accsincelast, rewardsincelast, acceptsincelast, pundersincelast = [ 
                EasyPoissonBootstrapAcc(batch_size=batch_size) for _ in range(5) ]
            allrewardssincelast = []
            print(f'sampler.palgo = { 1/sampler.invpalgo }')
            
    evar = compute_evar_ci(np.concatenate(allrewards, axis=None), q)
    evarsincelast = compute_evar_ci(np.concatenate(allrewardssincelast, axis=None), q)
    print('{:<5d}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<17s}\t{:<10.5f}'.format(
            avloss.n, avloss.formatci(), sincelast.formatci(), acc.formatci(),
            accsincelast.formatci(), f'[{evar[0]:.4f},{evar[1]:.4f}]', f'[{evarsincelast[0]:.4f},{evarsincelast[1]:.4f}]',
            punder.formatci(), pundersincelast.formatci(),
            accept.formatci(), acceptsincelast.formatci(),
            time.time() - start),
          flush=True)
    print(f'sampler.palgo = { 1/sampler.invpalgo }')
    print(f'0.2: {compute_evar_ci(np.concatenate(allrewards, axis=None), 0.2)} 0.5: {compute_evar_ci(np.concatenate(allrewards, axis=None), 0.5)}')

import sys
approxargmax = True if int(sys.argv[3]) > 0 else False
q = float(sys.argv[2])
mydata = ArffToPytorch(sys.argv[1], target='price', skipcol=['id', 'address'], skiprow=lambda z: z['price'] > 1e6)

def flass():
    import random
    for initlr, tzero, eta, gammamin, gammamax, nalgos in (
       (
	   10**(-3 + 2 * random.random()),
	   1 + 10 * random.random(),
	   10**(-1 + random.random()),
	   (1 << random.randint(0, 5)),
	   (1 << random.randint(6, 12)),
	   random.randint(4, 8),
       )
       for _ in range(59)
    ):
        print('hypers ', initlr, tzero, eta, gammamin, gammamax, nalgos)
        learnOnline(mydata, q=q, seed=4545, initlr=initlr, tzero=tzero, batch_size=8, 
                    eta=eta, gammamin=gammamin, gammamax=gammamax, nalgos=nalgos,
                    modelfactory=lambda x: CauchyTruncatedNormal(dobs=x.shape[1], numrff=1024, sigma=1/10, device='cpu', approxargmax=approxargmax))

flass()
