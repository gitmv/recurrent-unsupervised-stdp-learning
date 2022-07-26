from Behaviour_Modules import *
from UI.UI_Helper import *
from Behaviour_Bar_Activator import *

ui = True
neuron_count = 2000
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

net = Network(tag='Cluster Formation Network')

#different h parameters for experiment
#target_activity = 0.05
#target_activity = 0.0125
target_activity = 0.00625

exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)


class Out(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity>0.0
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=50, height=20, depth=1), color=yellow, behaviour={

    #patterns different frequency
    10: Line_Patterns(center_x=50, center_y=[0,1,1,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], degree=0, line_length=100, random_order=True),

    #patterns same frequency
    #10: Line_Patterns(center_x=50, center_y=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], degree=0, line_length=100, random_order=True),

    50: Out()
})

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007), #0.02

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'

})

SynapseGroup(net=net, tag='ES,GLU', src='inp_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='SE,GLU', src='exc_neurons', dst='inp_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EE,GLU', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

class CWP(Classifier_base):
    def get_data_matrix(self, neurons):
        return get_partitioned_synapse_matrix(neurons, 'ES', 'W').T

CWP(net['exc_neurons', 0])

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm)
else:
    print('please use UI')
