import pandas as pd
import numpy as np
from dowhy import CausalModel
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import statsmodels.api as sm
#from contextlib import redirect_stdout


### Read data
data = pd.read_csv('data.csv', header=0, skiprows=[3943])
cols_all = data.columns.values


### Impute missing values
for col in cols_all:
    data[col] = data[col].replace('?', 0).astype(np.int64)


### Remove unwanted attributes
colnames_to_drop = ['nr'] + [col for col in cols_all if '_legid' in col]
data = data.drop(colnames_to_drop, axis=1)
cols = data.columns.values


### Add new attributes

# Maximum incoming hops
data['max_i_hops'] = np.max(data[[c for c in cols if c.startswith('i') and c.endswith('_hops')]], axis=1)

# Difference predicted - effective
x = []
for col in cols:
    if col.endswith('p'):
        data[col[:-1]+'diff'] = data[col]-data[col[:-1]+'e']
        x.append(data[col[:-1]+'diff'] >= 0)

# Difference within leg
max_leg_predicted = []
max_leg_effective = []
for l in ['i1','i2','i3','o']:
    cols_leg = [col for col in cols if col.startswith(l)]
    cols_leg_p = [c for c in cols_leg if c.endswith('p')]
    cols_leg_e = [c for c in cols_leg if c.endswith('e')]

    leg_predicted = np.sum(data[cols_leg_p], axis=1)
    leg_effective = np.sum(data[cols_leg_e], axis=1)
    for i,c in enumerate(data[cols_leg_p].columns.values):
        data[c[:-2]+'_in_time'] = (data[c].subtract(data[c[:-2]+'_e']) >= 0).replace(True, 1).replace(False, 0).astype(np.int64)

    max_leg_predicted.append(leg_predicted)
    max_leg_effective.append(leg_effective)
    data[l+'_time_diff'] = (leg_predicted - leg_effective)
    data[l+'_in_time'] = ((leg_predicted - leg_effective)>= 0).replace(True, 1).replace(False, 0).astype(np.int64)

# Total difference
max_leg_predicted = pd.concat(max_leg_predicted, axis=1)
max_leg_effective = pd.concat(max_leg_effective, axis=1)
ttd = (np.max(max_leg_predicted.iloc[:,:3], axis=1) + max_leg_predicted.iloc[:,3]) - (np.max(max_leg_effective.iloc[:,:3], axis=1) + max_leg_effective.iloc[:,3])
data['total_time_diff'] = ttd
data['in_time'] = (ttd >= 0).replace(True, 1).replace(False, 0).astype(np.int64)

used_cols = ['i1_time_diff', 'i2_time_diff', 'i3_time_diff']+[*data.columns.values[-11:]]
used_cols.insert(-3, 'o_hops')
used_cols.insert(3, 'max_i_hops')


### PC Algorithm
cg = pc(data.loc[:, used_cols].to_numpy(), indep_test="fisherz", uc_rule=1, show_progress=True)
cg.draw_pydot_graph(labels=used_cols)


### Add Process Model
nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()

# Incoming legs parallel
for i in range(3):
    for j in range(3):
        bk.add_forbidden_by_node(nodes[i], nodes[j])

# max_i_hops causes i_time_diff and in_time/total_time_diff
for i in range(3):
    bk.add_required_by_node(nodes[3], nodes[i])
    bk.add_forbidden_by_node(nodes[i], nodes[3])
bk.add_required_by_node(nodes[3], nodes[-2])
bk.add_required_by_node(nodes[3], nodes[-1])

# No incoming edges to max_i_hops and legs
for i in range(len(used_cols)): 
    bk.add_forbidden_by_node(nodes[i], nodes[used_cols.index('max_i_hops')])
    
# in_time + total_time_diff no outgoing edges except tdf->it
for i,col in enumerate(used_cols[-2:]):
    for j in range(len(used_cols)-(i+1)):
        bk.add_forbidden_by_node(nodes[-(i+1)], nodes[j])


# hops determine values for 2nd/3rd hop (1st hop always exists), only for outgoing leg
for k in [c for c in used_cols if '_2' in c or '_3' in c]:
    bk.add_forbidden_by_node(nodes[used_cols.index(k)], nodes[used_cols.index('o_hops')])
    bk.add_required_by_node(nodes[used_cols.index('o_hops')], nodes[used_cols.index(k)])

# Take timely order into account
events = ['rcs', 'dep_1', 'rcf_1', 'dep_2', 'rcf_2', 'dep_3', 'rcf_3', 'dlv']
for ev in events[1:]:
    evs = [e for e in used_cols if ev in e] # All attributes belonging to event
    idx = used_cols.index(evs[0])   # All attributes of events before event evs
    for e in evs:
        for i in used_cols[:idx]:
            bk.add_forbidden_by_node(nodes[used_cols.index(e)], nodes[used_cols.index(i)])

# Manual assumptions
bk.add_forbidden_by_node(nodes[used_cols.index('max_i_hops')], nodes[used_cols.index('o_hops')])
bk.add_forbidden_by_node(nodes[used_cols.index('max_i_hops')], nodes[used_cols.index('o_rcf_1_in_time')])
bk.add_forbidden_by_node(nodes[used_cols.index('max_i_hops')], nodes[used_cols.index('o_dep_1_in_time')])
bk.add_forbidden_by_node(nodes[used_cols.index('o_time_diff')], nodes[used_cols.index('o_dlv_in_time')])
bk.add_required_by_node(nodes[used_cols.index('i2_time_diff')], nodes[used_cols.index('in_time')])
bk.add_required_by_node(nodes[used_cols.index('o_hops')], nodes[used_cols.index('in_time')])

cgbk = pc(data[used_cols].to_numpy(), indep_test="fisherz", uc_rule=1, show_progress=True, background_knowledge=bk)

# Visualization using pydot
cgbk.draw_pydot_graph(labels=used_cols)


### Make graph string from causal assumptions
graph = '{'
for c in used_cols: # Nodes
    graph += c + '; '
for i,c in enumerate(used_cols):    # Edges
    for j, cc in enumerate(used_cols):
        if i < j:
            print(c, cc, cgbk.G.graph[i,j], cgbk.G.graph[j,i])
            if cgbk.G.graph[i,j] == -1:
                graph += c + ' -> ' + cc + '; '
            elif cgbk.G.graph[i,j] == 1 and cgbk.G.graph[j,i] == 1:
                graph += c + ' -> ' + cc + '; '
graph += '}'
causal_graph = f""" digraph {graph} """


### Use DoWhy

# Define Causal Graph
model= CausalModel(
    data=data[used_cols],
    treatment=['o_hops', 'max_i_hops', 'o_in_time', 'i1_time_diff', 'i2_time_diff', 'i3_time_diff', 'o_dlv_in_time'],
    outcome="in_time",
    graph=causal_graph
)

# Identify the causal effect
estimands = model.identify_effect(proceed_when_unidentifiable=True)
#with open('dowhyOutput.txt', 'a+', encoding='utf-8') as f:
 #   with redirect_stdout(f):
print(estimands)

# Causal Effect Estimation
estimate = model.estimate_effect(estimands,
                                        method_name="backdoor.generalized_linear_model",
                                       confidence_intervals=False,
                                       test_significance=True,
                                        method_params = {
                                            'num_null_simulations':10,
                                            'num_simulations':10,
                                            'num_quantiles_to_discretize_cont_cols':10,
                                            'fit_method': "statsmodels",
                                            'glm_family': sm.families.Binomial(), # logistic regression
                                            'need_conditional_estimates':False
                                        },
                                       )
#with open('dowhyOutput.txt', 'a+', encoding='utf-8') as f:
#    with redirect_stdout(f):
print(estimate)
print("Causal Estimate is " + str(estimate.value))

# Refute the obtained estimate using robustness checks.
refute_results = model.refute_estimate(estimands, estimate, method_name="random_common_cause")
#with open('dowhyOutput.txt', 'a+', encoding='utf-8') as f:
 #   with redirect_stdout(f):
print(refute_results)
