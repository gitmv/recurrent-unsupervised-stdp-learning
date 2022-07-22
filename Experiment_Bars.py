from Behaviour_Modules import *
from Helper import *
from UI.UI_Helper import *
from Behaviour_Bar_Activator import *

ui = True
neuron_count = 1000#1500#2400
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

net = Network(tag='Bar Learning Network')

target_activity = 0.10#0.02#0.1
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

#20 10
#28*3, height=28*2
NeuronGroup(net=net, tag='exc_neurons', size=NeuronDimension(width=28*2, height=28*2, depth=1), color=blue, behaviour={#60 30get_squared_dim(neuron_count)

    # excitatory input
    #10: Line_Patterns(center_x=5, center_y=[0,1,2,3,4,5,6,7,8,9], degree=0, line_length=10, strength=5.0, random_order=False),
    10: Line_Patterns(center_x=5, center_y=5, degree=list(np.arange(0, 360, 20)), line_length=10, strength=5.0, random_order=False),
    #10: MNIST_Patterns_On_Off(center_x=14, class_ids=[10,11,12,13,14], patterns_per_class=1),

    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007), #0.02 '[0.03#TA]'

    # learning#0.5226654296858209
    #40: Learning_Inhibition_test(transmitter='GABA', a='[0.5#a]', b=0.0, c='[50.0#c]', d='[1.5#d]'),#'[0.6410769611853464#a]'
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(transmitter='GLU', strength='[0.005#S]'),#
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=1),##################
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=1),##################

    # output
    50: Generate_Output(exp=exc_output_exponent),

    90: Recorder(variables=['np.mean(n.output)'])
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(net['exc_neurons',0].size*0.2), color=red, behaviour={#neuron_count/10

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'

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
    generate_response_images(net)

