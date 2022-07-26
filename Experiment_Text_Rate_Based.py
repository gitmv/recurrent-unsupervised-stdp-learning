from Helper import *
from UI.UI_Helper import *
from Behaviour_Modules import *
from Behaviour_Text_Modules import *

class Generate_Output_Analog(Generate_Output):

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = np.clip(self.activation_function(neurons.activity),0,1)
        neurons._activity = neurons.activity.copy() #for plotting
        neurons.activity.fill(0)

class Generate_Output_Inh_Analog(Generate_Output_Inh):

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.output = np.clip(self.activation_function(self.avg_act),0,1)
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


ui = True
neuron_count = 1400

input_steps = 30000
recovery_steps = 10000
free_steps = 5000

grammar = get_random_sentences(2)

input_density=0.92
target_activity = 1.0 / len(''.join(grammar))
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

net = Network(tag='(Rate Based) Text Learning Network')

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    # excitatory input
    10: Text_Generator(text_blocks=grammar),
    11: Text_Activator(input_density=input_density, strength=1.0),
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
    50: Generate_Output_Analog(exp=exc_output_exponent),

    # reconstruction
    80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh_Analog(slope=inh_output_slope, duration=2),

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
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=sm)
