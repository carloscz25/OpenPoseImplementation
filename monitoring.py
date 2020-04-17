import pickle
import torch
from  torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


def allmodules(model, d, currname=''):
    for name, module in model._modules.items():
        d[currname + "_" + name] = module
        allmodules(module, d, currname + "_" + name)

def add_scalars_for_modname(writer, mods, name, step):
    d = {}
    for m in mods.keys():
        try:
            if m.index(name) >= 0:
                mod = mods[m]
                minp, maxp = torch.min(mod._parameters['weight']), torch.max(mod._parameters['weight'])
                d[m+'_minP_'] = minp.item()
                d[m+'_maxP_'] = maxp.item()
                minb, maxb = torch.min(mod._parameters['bias']), torch.max(mod._parameters['bias'])
                d[m + '_minB_'] = minb.item()
                d[m + '_maxB_'] = maxb.item()
        except:
            pass
    writer.add_scalars(name, d, step)

def monitor_gradient_shift(writer, mods, names, step, filepath):
    d_old = {}
    d = {}
    firstpass = False
    if os.path.exists(filepath) == True:
        d_old = pickle.load(open(filepath, 'rb'))
    else:
        firstpass = True
    for name in names:
        shiftsw = {}
        shiftsb = {}
        for m in mods.keys():
            try:
                if m.index(name) >= 0:
                    mod = mods[m]
                    d[m] = {}
                    if ('bias' in mod._parameters)==True:
                        hasbias=True
                    else:
                        hasbias=False
                    if mod._parameters['weight'].device.type == "cuda":
                        mw = mod._parameters['weight'].clone().to(torch.device("cpu")).detach().numpy()
                    else:
                        mw = mod._parameters['weight'].clone().detach().numpy()
                    if hasbias:
                        # mb = mod._parameters['bias'].clone().detach().numpy()
                        if mod._parameters['bias'].device.type == "cuda":
                            mb = mod._parameters['bias'].clone().to(torch.device("cpu")).detach().numpy()
                        else:
                            mb = mod._parameters['bias'].clone().detach().numpy()
                    mw[mw > 0] = 1
                    mw[mw == 0] = 0
                    mw[mw < 0] = -1
                    if hasbias:
                        mb[mb > 0] = 1
                        mb[mb == 0] = 0
                        mb[mb < 0] = -1
                    if firstpass==True:
                        d[m]['weight'] = mw
                        d[m]['bias'] = mb
                        pass
                    else:
                        mow = d_old[m]['weight']
                        if hasbias:
                            mob = d_old[m]['bias']
                        diffw = mw - mow
                        if hasbias:
                            diffb = mb - mob

                        d[m]['weight'] = mw
                        if hasbias:
                            d[m]['bias'] = mb
                        # counting shifts
                        ds = d[m]
                        shiftsw[m] = (np.count_nonzero(diffw) / np.size(diffw))
                        shiftsb[m] = (np.count_nonzero(diffb) / np.size(diffb))
            except:
                pass
        if firstpass==False:
            writer.add_scalars(name+'gradweights', shiftsw, step)
            writer.add_scalars(name + 'gradbias', shiftsb, step)
    pickle.dump(d, open(filepath, 'wb'))

def monitor_gradient_mean(writer, mods, names, step):
    for m in mods.keys():
        d = {}
        for name in names:
            if indexof(m, name) >=0:
                mod = mods[m]
                avgw = np.average(mod._parameters['weight'].grad.numpy())
                d[m + '_weightgradmean']= avgw
                if module_has_bias(mod):
                    avgb = np.average(mod._parameters['bias'].grad.numpy())
                    d[m + '_biasgradmean'] = avgb
            writer.add_scalars(name, d, step)

def monitor_parameter_mean(writer, mods, names, step):
    for m in mods.keys():
        d = {}
        for name in names:
            if indexof(m, name) >=0:
                mod = mods[m]
                avgw = np.average(mod._parameters['weight'].clone().detach().numpy())
                d[m + '_weightmean']= avgw
                if module_has_bias(mod):
                    avgb = np.average(mod._parameters['bias'].clone().detach().numpy())
                    d[m + '_biasmean'] = avgb
            writer.add_scalars(name, d, step)

def module_has_bias(m):
    try:
        b = m._parameters['bias']
        return True
    except:
        return False

def indexof(strcontainer, strcontained):
    try:
        return strcontainer.index(strcontained)
    except:
        return -1

