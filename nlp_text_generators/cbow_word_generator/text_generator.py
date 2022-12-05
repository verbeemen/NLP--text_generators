import os
import torch


class Model(torch.nn.Module):
    def __init__(self, seed) -> None:
        super().__init__()

        # generator
        self.g = torch.Generator().manual_seed(seed)

        # embedding layer
        self.emb = torch.randn((27, 2), dtype=torch.float32, requires_grad=True, generator=self.g)

        # first layer
        self.w1 = torch.randn((6, 100), dtype=torch.float32, requires_grad=True, generator=self.g)
        self.b1 = torch.randn((100,), dtype=torch.float32, requires_grad=True, generator=self.g)

        # second layer
        self.w2 = torch.randn((100, 27), dtype=torch.float32, requires_grad=True, generator=self.g)
        self.b2 = torch.randn((27,), dtype=torch.float32, requires_grad=True, generator=self.g)

        # list of all parameters
        self.parameters = [self.emb, self.w1, self.b1, self.w2, self.b2]


    def __call__(self, x):

        emb = self.emb[x].view(-1, self.emb.shape[1] * x.shape[-1])
        inner_prod = emb @ self.w1 + self.b1
        h = torch.tanh(inner_prod)
        return h @ self.w2 + self.b2

class CBOWGenerator:
    def __init__(
        self, n: int = 3, path_corpus: str = "./data/names.csv", seed: int = 26071991, start_stop_token: str = "#"
    ):
        """ """

        # store the parameters
        self.ngram: int = self.validate_n(n)
        self.path_corpus: str = path_corpus
        corpus: list[str] = self.load_corpus(self.path_corpus)
        self.start_stop_token: str = start_stop_token

        # create the lookup tables
        self.dict_char_to_idx: dict[str:int] = {}
        self.dict_idx_to_char: dict[str:int] = {}
        self.set_dicts_by_ngram(corpus)

        # create training data
        X, y = self.create_training_data(corpus)

        # Load the model
        self.model = Model(seed)
        self.train_model(X, y)
        
        
    def set_path_corpus(self, path_corpus):
        """Set the path to the corpus.

        Args:
            @param: path_corpus (str): Path to the corpus.
        """
        self.path_corpus = path_corpus

    def validate_n(self, n: int) -> int:
        """Validate the ngram value.

        Args:
            @param: n (int): ngram value.
            @return: n (int): Validated ngram value.
        """

        # check if the ngram value is valid
        if n <= 0:
            raise ValueError(f"Invalid ngram value {n}, \nThe value must be greater than 0")

        elif n > 5:
            raise ValueError(f"N should be reasonably small, e.g. 1, 2, 3, 4, 5. \n")

        # return the validated ngram value
        return n

    def load_corpus(self, path_corpus) -> list[str]:
        """Load the corpus from a given path.

        Args:
            @param: path_corpus (str): Path to the corpus.
            @return: corpus (list[str]): List of items.
        """

        # check if the file exists
        if os.path.exists(path_corpus) is False:
            raise FileNotFoundError(f"File {path_corpus} not found")

        # load the corpus in lowercase
        data = open(path_corpus, "r").read().lower().splitlines()

        # Skip the header line
        if path_corpus.endswith(".csv"):
            data = data[1:]

        # return the corpus
        return data

    def set_dicts_by_ngram(self, corpus: list[str]):
        """create the lookup tables from the corpus

        Args:
            @param: corpus (list[str]): List of items.
            @set: dict_char_to_idx
            @set: dict_idx_to_char
        """

        # create the lookup tables # -> 0, a -> 1, b -> 2, c -> 3, ... , z -> 26
        chars = sorted(list(set("".join(corpus))))
        self.dict_char_to_idx = {self.start_stop_token: 0}
        self.dict_char_to_idx |= {
            char: idx for idx, char in enumerate(set(chars) - set(self.dict_char_to_idx), start=1)
        }

        # invert the lookup tables # -> 0, 1 -> a, 2 -> b, 3 -> c, ... , 26 -> z
        self.dict_idx_to_char = {idx: char for char, idx in self.dict_char_to_idx.items()}

    def create_training_data(self, corpus: list[str]):
        """Create the training data.

        Args:
            @param: corpus (list[str]): List of items.
            @set: X (torch.tensor): The training data.
            @set: y (torch.tensor): The labels.
        """

        # create the training data
        X, y = [], []
        for item in corpus:
            item = (self.start_stop_token * self.ngram) + item + self.start_stop_token
            for ngram, next_char in zip(zip(*[item[i:] for i in range(self.ngram)]), item[self.ngram :]):
                # add the values to x and y
                X.append([self.dict_char_to_idx[char] for char in ngram])
                y.append(self.dict_char_to_idx[next_char])

        # convert the lists to tensors
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.int64)


    
    def train_model(self, X, Y):

        for i in range(10_000):

            # take a minibatch
            ix = torch.randint(0, X.shape[0], (32,))


            #
            # Forward pass
            #

            # get the logits
            logits = self.model(X[ix,:])

            # Get probabilities
            # logits are not probabilities yet, so we need to apply softmax to get probabilities
            # In general the values of logits are roughly in the range of -3 to 3.
            # So when we apply en exponential function to them, the values will be in the range of 0 to +infinty.
            ## ...
            ## exp(-1) = 0.1353       
            ## exp(-0.001) = 0.999    
            ## exp( 0) = 1
            ## exp( 0.001) = 1.001
            ## exp( 1) = 2.718
            ## ...
            # basically, if we use the exp, every x >= 0 will have a value of 1 or more. 
            # and every < 0 will have a value of 1 or less (but still positive).
            # Last but not least, if we devide the exp(x) by the sum of all exp(x), we will get a probability distribution.
            
            # get the probabilities
            # counts = logits.exp()
            # prob = counts / counts.sum(1, keepdim=True)
            loss = torch.nn.functional.cross_entropy(logits, Y[ix])
         
            #
            # Backward pass
            #
            for param in self.model.parameters:
                param.grad = None
            loss.backward()

            # Update the parameters
            for param in self.model.parameters:
                param.data += -0.1 * param.grad
        else:
            # Final evaluation
            loss = torch.nn.functional.cross_entropy(self.model(X), Y)
            print(f"Loss: {loss.item():.4f}")

    def generate_text(self, start="") -> str:
        """Generate a text given a start string.

        Args:
            @param: start (str): Start string.
            @return: text (str): Generated text.
        """

        # start from:
        ngram = (self.start_stop_token * self.ngram + start.lower())[-self.ngram :]
        ngram = torch.tensor([self.dict_char_to_idx[char] for char in ngram], dtype=torch.int64)

        return self.model(ngram)


if __name__ == "__main__":

    # Import the necessary libraries
    from nlp_text_generators.cbow_word_generator.text_generator import CBOWGenerator

    # Initialize the generator
    generator = CBOWGenerator(n=3, path_corpus="./data/names.csv", seed=42)

    # # Show some results
    # print("--- Show Results ---")
    # for i in range(5):
    #     generator.generate_text()

    # print("--- Show Results ---")
    # for i in range(25):
    #     print(generator.generate_text(start="ma"))

    # # Evaluate the model
    # print("--- Evaluate Model ---")
    # generator.evaluate_model(items='dieter')
    # generator.evaluate_model(items=['bruno'])
    # import numpy as np
    # generator.evaluate_model(items=np.array(['dieter']))
    # generator.evaluate_model()
