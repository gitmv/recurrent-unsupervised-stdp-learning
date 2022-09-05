from Helper import *
from UI.UI_Helper import *
from Behaviour_Core_Modules import *
from Behaviour_Text_Modules import *

ui = True
neuron_count = 2400

input_steps = 30000
recovery_steps = 10000
free_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

input_density=0.92
target_activity = 1.0 / len(''.join(grammar))
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

net = Network(tag='Text Learning Network')

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    #9: Exception_Activator(), # use for manual text input with GUI code tab...

    # excitatory input
    10: Text_Generator(text_blocks=grammar),
    11: Text_Activator(input_density=input_density, strength=1.0),#remove for non input tests
    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent), #'[0.614#EXP]'

    # reconstruction
    80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

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

#User interface
if __name__ == '__main__' and ui:
    add_all_analysis_modules(net['exc_neurons', 0])
    show_UI(net, sm)
else:
    net.add_behaviours_to_object({200: Recorder(variables=['np.mean(n.output)'])}, net.exc_neurons)
    train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=sm)
    plot_output_trace(net['np.mean(n.output)', 0], input_steps, recovery_steps, net.exc_neurons.target_activity)
