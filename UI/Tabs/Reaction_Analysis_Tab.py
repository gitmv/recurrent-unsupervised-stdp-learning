from PymoNNto.Exploration.Network_UI.TabBase import *


class Reaction_Analysis_Tab(TabBase):

    def __init__(self, title='Reaction_analysis'):
        super().__init__(title)

    def add_recorder_variables(self, neuron_group, Network_UI):
        return

    def initialize(self, Network_UI):
        if Network_UI.network['Activity_Response_Behaviour', 0] and Network_UI.network['Text_Generator', 0] is not None:
            self.reconstruction_tab = Network_UI.Next_Tab(self.title)

            self.grid = QGridLayout()
            self.grid.setAlignment(Qt.AlignLeft)
            Network_UI.current_H_block.addLayout(self.grid)
            Network_UI.current_H_block.setAlignment(Qt.AlignTop)

            generator = Network_UI.network['Text_Generator', 0]
            self.labels = []

            for y, char in enumerate(generator.alphabet):
                self.labels.append([])
                for timestep in range(11):
                    label = QLabel('<font color='+('#%02x%02x%02x'% (timestep*25,timestep*25,timestep*25)).upper()+'>'+char.replace(' ','_')+'</font>')
                    label.char=char
                    self.labels[-1].append(label)
                    font = label.font()
                    font.setPointSizeF(12-len(generator.alphabet)/10)
                    label.setFont(font)
                    self.grid.addWidget(label, y, timestep)

            self.img = Network_UI.Add_Image_Item(False, False, title=' neuron rec')

            self.recon_text_label = QLabel()
            Network_UI.Add_element(self.recon_text_label)

            #Network_UI.Next_H_Block()

            self.net_grid = QGridLayout()
            self.net_grid.setAlignment(Qt.AlignLeft)
            Network_UI.current_H_block.addLayout(self.net_grid)
            Network_UI.current_H_block.setAlignment(Qt.AlignTop)


    def update(self, Network_UI):
        if Network_UI.network['Activity_Response_Behaviour', 0] and Network_UI.network['Text_Generator', 0] is not None and self.reconstruction_tab.isVisible():
            group = Network_UI.selected_neuron_group()
            nra = Network_UI.network['Activity_Response_Behaviour', 0]

            res=nra.get_reconstruction_matrix(Network_UI.selected_neuron_id())

            res = np.array(res)
            res = res-np.min(res)
            self.img.setImage(np.fliplr(res.copy()))

            for y in range(res.shape[0]):
                for x in range(res.shape[1]):
                    m = (np.max(res)+np.max(res[y, :]))/2.0
                    if m == 0:
                        m = 1.0
                    val = 255-np.clip(int((res[y, x]-(np.mean(res[y, :])/2.0))/m*255.0), 0, None)
                    self.labels[x][y].setText('<font color='+('#%02x%02x%02x' % (val, val, val)).upper()+'>'+self.labels[x][y].char.replace(' ','_')+'</font>')

            self.recon_text_label.setText(nra.get_representation(Network_UI.selected_neuron_id()))
