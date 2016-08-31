from random import choice, randrange
import matplotlib.pyplot as plt
import pycnn as pc

EOS = "<EOS>"  # all strings will end with the End Of String token
characters = list("abcd")
characters.append(EOS)

int2char = list(characters)
char2int = {c: i for i, c in enumerate(characters)}

VOCAB_SIZE = len(characters)


def sample_model(min_length, max_lenth):
    random_length = randrange(min_length, max_lenth)  # Pick a random length
    random_char_list = [choice(characters[:-1]) for _ in xrange(random_length)]  # Pick random chars
    random_string = ''.join(random_char_list)
    return random_string, random_string[::-1]  # Return the random string and its reverse


def train(network, train_set, val_set, epochs=10):
    def get_val_set_loss(network, val_set):
        loss = [network.get_loss(input_string, output_string).value()
                for input_string, output_string in val_set]
        return sum(loss)

    train_set = train_set * epochs
    trainer = pc.SimpleSGDTrainer(network.model)
    losses = []
    iterations = []
    for i, training_example in enumerate(train_set):
        input_string, output_string = training_example

        loss = network.get_loss(input_string, output_string)
        loss_value = loss.value()
        loss.backward()
        trainer.update()

        # Accumulate average losses over training to plot
        if i % (len(train_set) / 100) == 0:
            val_loss = get_val_set_loss(network, val_set)
            losses.append(val_loss)
            iterations.append(i / ((len(train_set) / 100)))
            print '{} validation check: {}'.format(i, val_loss)

    plt.plot(iterations, losses)
    plt.axis([0, 100, 0, len(val_set) * MAX_STRING_LEN])
    plt.show()
    print 'Final loss on validation set:', get_val_set_loss(network, val_set)


class SimpleRNNNetwork:
    def __init__(self, rnn_num_of_layers, embeddings_size, state_size):
        self.model = pc.Model()
        # the embedding paramaters
        self.model.add_lookup_parameters("lookup", (VOCAB_SIZE, embeddings_size))
        # the rnn
        RNN_BUILDER = pc.LSTMBuilder
        self.RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)
        # project the rnn output to a vector of VOCAB_SIZE length
        self.model.add_parameters("output_w", (VOCAB_SIZE, state_size))
        self.model.add_parameters("output_b", (VOCAB_SIZE))

    def _add_eos(self, string):
        string = list(string) + [EOS]
        return [char2int[c] for c in string]

    # preprocessing function for all inputs (should be overriden for different problems)
    def _preprocess_input(self, string):
        return self._add_eos(string)

    # preprocessing function for all outputs (should be overriden for different problems)
    def _preprocess_output(self, string):
        return self._add_eos(string)

    def _embed_string(self, string):
        lookup = self.model["lookup"]
        return [lookup[char] for char in string]

    def _run_rnn(self, init_state, input_vecs):
        s = init_state
        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs

    def _get_probs(self, rnn_output):
        output_w = pc.parameter(self.model["output_w"])
        output_b = pc.parameter(self.model["output_b"])
        probs = pc.softmax(output_w * rnn_output + output_b)
        return probs

    def get_loss(self, input_string, output_string):
        input_string = self._preprocess_input(input_string)
        output_string = self._preprocess_output(output_string)
        pc.renew_cg()
        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)
        loss = []
        for rnn_output, output_char in zip(rnn_outputs, output_string):
            probs = self._get_probs(rnn_output)
            loss.append(-pc.log(pc.pick(probs, output_char)))
        loss = pc.esum(loss)
        return loss

    def _predict(self, probs):
        probs = probs.value()
        predicted_char = int2char[probs.index(max(probs))]
        return predicted_char

    def generate(self, input_string):
        input_string = self._preprocess_input(input_string)
        pc.renew_cg()

        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)

        output_string = []
        for rnn_output in rnn_outputs:
            probs = self._get_probs(rnn_output)
            predicted_char = self._predict(probs)
            output_string.append(predicted_char)
        output_string = ''.join(output_string)
        return output_string


if __name__ == "__main__":
    print sample_model(4, 5)
    print sample_model(5, 10)

    MAX_STRING_LEN = 15

    train_set = [sample_model(1, MAX_STRING_LEN) for _ in xrange(3000)]
    val_set = [sample_model(1, MAX_STRING_LEN) for _ in xrange(50)]

    RNN_NUM_OF_LAYERS = 2
    EMBEDDINGS_SIZE = 4
    STATE_SIZE = 128

    rnn = SimpleRNNNetwork(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE)
    train(rnn, train_set, val_set)
    print rnn.generate('ab')
