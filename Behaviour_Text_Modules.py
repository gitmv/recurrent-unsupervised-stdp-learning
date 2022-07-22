from PymoNNto import *
import random
import random

class Text_Generator(Behaviour):

    set_variables_on_init = True

    def get_text_blocks(self):
        return self.get_init_attr('text_blocks', [])

    def set_variables(self, neurons):

        self.text_blocks = self.get_text_blocks()
        self.current_block_index = -1
        self.position_in_current_block = -1
        self.alphabet = unique(''.join(self.text_blocks)) #list not string!!!!
        self.history = ''

        # output
        neurons.current_char = ''
        neurons.current_char_index = -1

        # manual activation
        self.next_char = None

        if self.get_init_attr('set_network_size_to_alphabet_size', False):
            dim = get_squared_dim(len(self.alphabet)) #NeuronDimension
            dim.set_variables(neurons) #set size and x, y, z, width, height, depth

        char_count_vec = self.count_chars_in_blocks()
        self.char_weighting = char_count_vec / np.mean(char_count_vec)


    def new_iteration(self, neurons):

        if self.next_char is not None:# manual activation
            neurons.current_char = self.next_char
            self.next_char = None
        else:
            neurons.current_char = self.get_char(next=True)

        neurons.current_char_index = self.char_to_index(neurons.current_char)

        self.history += neurons.current_char

    def get_char(self, next=False):
        if next:
            self.position_in_current_block += 1

        if self.position_in_current_block >= len(self.text_blocks[self.current_block_index]):
            self.current_block_index=self.get_next_block_index()
            self.position_in_current_block = 0

        return self.text_blocks[self.current_block_index][self.position_in_current_block]

    def index_to_char(self, index):
        return self.alphabet[index]

    def char_to_index(self, char):
        return self.alphabet.index(char)

    def get_next_block_index(self):
        return np.random.randint(len(self.text_blocks))

    def set_next_char(self, char):#manual activation
        self.next_char = char

    def get_words(self):
        return unique([word for word in ' '.join(self.text_blocks).replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').split(' ') if word != ''])

    def count_chars_in_blocks(self):
        result = np.zeros(len(self.alphabet))
        for block in self.text_blocks:
            for c in block:
                result[self.char_to_index(c)] += 1
        return result

    def get_text_score(self, text):
        block_scores = [1 for _ in self.text_blocks]
        for i in range(len(text)):
            for bi, block in enumerate(self.text_blocks):
                block_score = 0
                comp_text = text[i:i+len(block)]
                comp_block = block[0:len(comp_text)]
                for t, b in zip(comp_text, comp_block):
                    if t == b:
                        block_score += 1 * (1/self.char_weighting[self.char_to_index(t)])

                block_scores[bi] += (block_score*block_score)/len(text)
        score = 0
        for bs in block_scores:
            score += np.sqrt(bs)
        return score

class Text_Activator(Behaviour):

    def set_variables(self, neurons):
        self.text_generator = neurons['Text_Generator', 0]

        input_density = self.get_init_attr('input_density', 0.5)
        activation_size = np.floor((neurons.size * input_density) / len(self.text_generator.alphabet)) #average char cluster size

        neurons.mean_network_activity = activation_size / neurons.size  # optional/ can be used by other (homeostatic) modules

        if self.get_init_attr('char_weighting', True):
            cw = self.text_generator.char_weighting
        else:
            cw = None

        neurons.Input_Weights = self.one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, cw)
        neurons.Input_Mask = np.sum(neurons.Input_Weights, axis=1) > 0

        neurons.input_grammar = neurons.get_neuron_vec()

        self.strength = self.get_init_attr('strength', 1, neurons)

    def new_iteration(self, neurons):
        neurons.input_grammar = neurons.Input_Weights[:, neurons.current_char_index].copy()*self.strength
        neurons.activity += neurons.input_grammar

    def one_hot_vec_to_neuron_mat(self, input_size, output_size, activation_size, input_weighting=None):

        result = np.zeros((output_size, input_size))

        available = set(range(output_size))

        for a in range(input_size):

            char_activiation_size = activation_size

            if input_weighting is not None:
                char_activiation_size = activation_size * input_weighting[a]

            temp = random.sample(available, int(char_activiation_size))
            result[temp, a] = 1
            available = set([n for n in available if n not in temp])
            assert len(available) > 0, 'Input too big for non-overlapping neurons'

        return result

class Text_Reconstructor(Behaviour):

    def set_variables(self, neurons):
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if neurons['Text_Activator', 0] is not None:
            char_act = neurons.Input_Weights.transpose().dot(neurons.output)

            self.current_reconstruction_char_index = np.argmax(char_act)
            self.current_reconstruction_char = neurons['Text_Activator', 0].text_generator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char

def unique(l):
    return list(sorted(set(l)))

def get_random_sentences(n_sentences):
    sentences = [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.', ' parrots can fly.', 'the fish swims']
    return sentences[0:n_sentences]

def get_char_sequence(n_chars):
    sequence = '. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[](){}<>'
    return [sequence[0:n_chars]]

def get_long_text():
    return [' fox eats meat. boy drinks juice. penguin likes ice.']