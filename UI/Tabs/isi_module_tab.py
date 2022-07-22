from PymoNNto.Exploration.Network_UI.TabBase import *
from PymoNNto.Exploration.Visualization.Visualization_Helper import *

class isi_module_tab(TabBase):

    def initialize(self, Network_UI):
        self.isi_tab = Network_UI.Next_Tab(self.title)

        self.cruves = Network_UI.Add_plot_curve(number_of_curves=2, x_label='inter spike interval', y_label='approximated frequency')
        self.inh_cruve = Network_UI.Add_plot_curve(number_of_curves=1, x_label='inter spike interval inhibition', y_label='approximated frequency')

    def update(self, Network_UI):
        if self.isi_tab.isVisible():

            group = Network_UI.selected_neuron_group()

            if len(group['isi_reaction_module']) > 0:
                self.cruves[0].setData(list(range(100)), group['isi_reaction_module', 0].isi_history[Network_UI.selected_neuron_id()])
                self.cruves[1].setData(list(range(100)), group['isi_reaction_module', 0].target_distribution)
                self.inh_cruve.setData(list(range(100)), group['isi_reaction_module', 0].isi_inhibition[Network_UI.selected_neuron_id()])
