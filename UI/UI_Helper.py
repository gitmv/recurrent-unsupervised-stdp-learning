from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
from PymoNNto.Exploration.AnalysisModules import *
from UI.Tabs.Reaction_Analysis_Tab import *
from UI.Analysis_Modules import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)
black = (0.0, 0.0, 0.0, 255.0)

def show_UI(net, sm, qa=['STDP', 'Text_Activator'], additional_modules=None):

    # create ui tab dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['output'], quick_access_tags=qa),#['STDP', 'Text_Activator', 'Input', 'ff', 'fb']
        get_my_default_UI_modules(),
        Reaction_Analysis_Tab(),
        additional_modules
    )

    #modify some tabs
    my_modules[multi_group_plot_tab].__init__(['output|target_activity|0.0|target_activity*2', '_activity', 'sensitivity', 'linh'])
    my_modules[single_group_plot_tab].__init__(['output', '_activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity', 'weight_norm_factor'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons')

    #create classification AnalysisModules to classify characters and input-non-input neuron classification
    neurons = net['exc_neurons', 0]

    if hasattr(neurons, 'Input_Weights'):
        char_classes = np.sum((neurons.Input_Weights>0) * np.arange(1,neurons.Input_Weights.shape[1]+1,1), axis=1).transpose()#neurons.Input_Weights.shape[1]
        Static_Classification(parent=neurons, name='char', classes=char_classes)

    if hasattr(neurons, 'Input_Mask'):
        Static_Classification(parent=neurons, name='input class', classes=neurons.Input_Mask)

    # launch ui
    Network_UI(net, modules=my_modules, label=net.tags[0], storage_manager=sm, group_display_count=len(net.NeuronGroups), reduced_layout=False).show()



