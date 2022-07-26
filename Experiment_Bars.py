from Behaviour_Modules import *
from Helper import *
from UI.UI_Helper import *
from Behaviour_Bar_Activator import *

ui = True
input_steps = 30000
recovery_steps = 10000
free_steps = 5000

net = Network(tag='Bar Learning Network')

target_activity = 0.10
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

#20 10
NeuronGroup(net=net, tag='exc_neurons', size=NeuronDimension(width=20, height=10, depth=1), color=blue, behaviour={

    # excitatory input
    #10: Line_Patterns(center_x=5, center_y=[0,1,2,3,4,5,6,7,8,9], degree=0, line_length=10, strength=5.0, random_order=False), #Experiment E
    10: Line_Patterns(center_x=5, center_y=5, degree=list(np.arange(0, 360, 20)), line_length=10, strength=5.0, random_order=False), #Experiment F

    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: STDP(transmitter='GLU', strength='[0.005#S]'),#
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=1),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=1),

    # output
    50: Generate_Output(exp=exc_output_exponent),

    90: Recorder(variables=['np.mean(n.output)'])
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(net['exc_neurons',0].size*0.2), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2),

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

net.sm = sm

Classifier_Weights_Pre(net['exc_neurons', 0]),
Classifier_Weights_Post(net['exc_neurons', 0]),

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm)
else:
    generate_response_images(net, input_steps, recovery_steps, free_steps)

