import os
import torch


class NGramTextGenerator:
    def __init__(self, n: int = 3, path_corpus: str = "./data/names.csv", seed: int = 26071991, start_stop_token: str = "#"):
        """A simple model that generates text, given a corpus of words or sentences.
            Basically it will select the next highest possible character based on the probability distribution of a given ngram.
            For example if the model is initialized to n = 3 (and we thus use a tri-gram),
            then the model will select the next possible character based on the known probability distribution of the previous 3 characters of E.g. 'abc' --> 'd'
            The probability distribution is calculated via a corpus.

        Note: This model is a spin-off of the youtube video from: Andrej Karpathy
                                                           title: The spelled-out intro to language modeling: building makemore
                                                             url: https://www.youtube.com/watch?v=PaCmpygFfXo

        Args:
            n (int, optional): ngram size. Defaults 3. (but can be 1, 2, 3, 4)
                              (amount of characters to consider for predicting the next character)
            path_corpus (str, optional): path to a corpus file. Defaults: 'data/names.csv'.
            seed (int, optional): a random seed.
        """

        # store the parameters
        self.ngram: int = self.validate_n(n)
        corpus: list[str] = self.load_corpus(path_corpus)
        self.start_stop_token: str = start_stop_token

        # create the lookup tables
        self.dict_ngram_to_idx = {}
        self.dict_idx_to_ngram = {}
        self.set_dicts_ngram(corpus)

        self.dict_next_char_to_idx = {}
        self.dict_idx_to_next_char = {}
        self.set_dicts_next_char(corpus)

        # Create the probability matrix
        self.probability_matrix = torch.zeros(
            (len(self.dict_ngram_to_idx), len(self.dict_next_char_to_idx)), dtype=torch.float32
        )
        self.fill_probability_matrix(corpus)

        # set the seed
        self.g = torch.Generator().manual_seed(seed)

    def validate_n(self, n: int) -> int:
        """Validate the ngram value.

        Args:
            @param: n (int): ngram value.
            @return: n (int): Validated ngram value.
        """

        # check if the ngram value is valid
        if n <= 0:
            raise ValueError(f"Invalid ngram value {n}, \nThe value must be greater than 0")

        if n not in [1, 2, 3, 4]:
            raise ValueError(
                f"ngram value {n} is not valid,\nworst-case ngram with value {n} will create a matrix of {(27**n)} by {(27)}"
            )

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

    def set_dicts_next_char(self, corpus: list[str]):
        """Set the lookup tables from the corpus,
           for the next character.

           Goal: given a ngram (uni, bi, tri, quad), select the next character.

        Args:
            @param: corpus (list[str]): List of items.
            @set: self.dict_next_char_to_idx
            @set: self.dict_idx_to_next_char
        """

        # create the lookup tables a -> 1, b -> 2, c -> 3, ... , z -> 26
        chars = sorted(list(set("".join(corpus))))
        self.dict_next_char_to_idx = {char: idx for idx, char in enumerate(chars, start=1)}

        # add the special character '#' to indicate the start and end of the item
        self.dict_next_char_to_idx[self.start_stop_token] = 0

        # reverse the lookup tables
        self.dict_idx_to_next_char = {idx: ngram for ngram, idx in self.dict_next_char_to_idx.items()}

    def set_dicts_ngram(self, corpus: list[str]):
        """Set the lookup tables from the corpus,
           for the given ngrams.

           Goal: given a ngram (uni, bi, tri, quad), select the next character.

        Args:
            @param: corpus (list[str]): List of names.
            @set: self.dict_next_char_to_idx
            @set: self.dict_idx_to_next_char
        """

        # join the corpus with the special character '#' * ngram
        corpus = self.start_stop_token * self.ngram + f'{self.start_stop_token * self.ngram}'.join(corpus) + self.start_stop_token * self.ngram
        tokens = sorted(list({"".join(ngram) for ngram in zip(*[corpus[i:] for i in range(self.ngram)])}))

        self.dict_ngram_to_idx = {char: idx for idx, char in enumerate(tokens)}

        # reverse the lookup tables
        self.dict_idx_to_ngram = {idx: ngram for ngram, idx in self.dict_ngram_to_idx.items()}

    def fill_probability_matrix(self, corpus: list[str]):
        """Fill the probability matrix with the probabilities of the next character.

        Args:
            @param: corpus (list[str]): List of items.
            @set: self.probability_matrix
        """

        for e, item_raw in enumerate(corpus):

            # add the special character '#' * ngram
            item = self.start_stop_token * self.ngram + item_raw + self.start_stop_token * self.ngram

            # get the ngrams and the next character
            for ngram, next_char in zip(zip(*[item[i:] for i in range(self.ngram)]), item[self.ngram :]):
                self.probability_matrix[
                    self.dict_ngram_to_idx["".join(ngram)], self.dict_next_char_to_idx[next_char]
                ] += 1

            print(f"[{e+1}/{len(corpus)}] - {item_raw}", end="\r")
        print("")

        # normalize the matrix
        self.probability_matrix = self.probability_matrix / self.probability_matrix.sum(dim=1, keepdim=True)

    def generate_text(self, start="") -> str:
        """Generate a text given a start string.

        Args:
            @param: start (str): Start string.
            @return: text (str): Generated text.
        """

        # collect the output
        output = list(start.lower())

        # start from:
        ngram = (self.start_stop_token * self.ngram + start)[-self.ngram :]
        start_stop = self.start_stop_token * self.ngram

        # generate
        while True:

            # get the index of the ngram
            ix = self.dict_ngram_to_idx[ngram]

            # select the probability distribution of the next character given the current ngram
            probabilities_next_char = self.probability_matrix[ix]

            # select the next character
            next_char_ix = torch.multinomial(
                probabilities_next_char, num_samples=1, replacement=True, generator=self.g
            ).item()

            # convert the index to the character
            next_char = self.dict_idx_to_next_char[next_char_ix]

            # check if the next_char is our stop character
            if next_char == start_stop[0]:
                break

            # update the ngram
            ngram = ngram[1:] + next_char

            # store the result
            output.append(next_char)


        return "".join(output)


if __name__ == "__main__":

    # Import the necessary libraries
    from n_gram_text_generator.generator import NGramTextGenerator

    # Initialize the generator
    generator = NGramTextGenerator(n=4, path_corpus="./data/names.csv", seed=42)

    # Show some results
    print("--- Show Results ---")
    for i in range(25):
        print(generator.generate_text())

    print("--- Show Results ---")
    for i in range(25):
        print(generator.generate_text(start="ma"))
