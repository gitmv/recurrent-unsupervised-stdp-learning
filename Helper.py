from PymoNNto.Exploration.Evolution.Interface_Functions import *
from PymoNNto import *

def train_and_generate_text(net, plastic_steps, recovery_steps=None, text_gen_steps=5000, sm=None, pretrained=False):

    net.simulate_iterations(plastic_steps, 100)

    # deactivate Input
    #net.deactivate_mechanisms('STDP')
    #net.deactivate_mechanisms('Normalization')
    net.deactivate_mechanisms('Text_Activator')

    # recovery phase
    if recovery_steps is not None:
        net.simulate_iterations(recovery_steps, 100)

    # text generation
    tr = net['Text_Reconstructor', 0]
    tr.reconstruction_history = ''
    net.simulate_iterations(text_gen_steps, 100)
    print(tr.reconstruction_history)

    # scoring
    score = net['Text_Generator', 0].get_text_score(tr.reconstruction_history)
    set_score(score, sm, info={'text': tr.reconstruction_history, 'simulated_iterations': net.iteration})


def plot_output_trace(data, plastic_steps, recovery_steps, target_activity, w=500, g=200):
    w2=int(w/2)

    plt.hlines(target_activity,-g, (w+g)*3+w+g, colors='k')

    plt.vlines((w+g)+w2, -0.1, 0.1, colors='k')
    plt.vlines((w+g)*2+w2, -0.1, 0.1, colors='k')

    plt.plot(np.arange(0, w), data[0:w])
    plt.plot(np.arange(0, w)+(w+g), data[plastic_steps-w2:plastic_steps+w2])
    plt.plot(np.arange(0, w)+(w+g)*2, data[plastic_steps+recovery_steps-w2:plastic_steps+recovery_steps+w2])
    plt.plot(np.arange(0, w)+(w+g)*3, data[-w:])
    plt.show()

def save_trace(it, net):
    neurons = net['exc_neurons', 0]
    a = net['np.mean(n.output)',0][-500:]

    max_a = neurons.target_activity * 2.0#np.maximum(np.max(a), neurons.target_activity * 2.0)
    min_a = 0.0#np.min(a)

    pps = 1  # pixels_per_step

    w = (len(a) - 1) * pps
    h = 100 * 5

    array = np.zeros([h, w, 3], dtype=np.uint8) + 255
    im = Image.fromarray(array, 'RGB')
    draw = ImageDraw.Draw(im)

    def f(y):
        return h - (y - min_a) / (max_a - min_a) * h

    temp = f(a[0])
    for i, b in enumerate(a[1:]):
        pos = f(b)
        draw.line((i * pps, temp, (i + 1) * pps, pos), fill=(0, 0, 0))
        temp = pos

    #draw.line((0, f(0), w, f(0)), fill=(255, 0, 0))
    draw.line((0, f(neurons.target_activity), w, f(neurons.target_activity)), fill=(0, 0, 255))
    #draw.line((0, f(neurons.target_activity * 2.0), w, f(neurons.target_activity * 2.0)), fill=(0, 255, 0))

    im.save(net.sm.absolute_path + "act"+ str(neurons.iteration) +".png")


def generate_response_images(net):
    neurons = net['exc_neurons', 0]
    neurons.add_analysis_module(Neuron_Classification_Colorizer())
    net.add_behaviours_to_object({100:Recorder(variables=['np.mean(n.output)'])}, neurons)

    net.simulate_iterations(30000, 501, batch_progress_update_func=save_trace)

    net.deactivate_mechanisms('STDP')
    net.deactivate_mechanisms('Normalization')
    net.deactivate_mechanisms('Input')

    net.simulate_iterations(5000, 201, batch_progress_update_func=save_trace)

    net.simulate_iterations(5000, 501, batch_progress_update_func=save_trace)