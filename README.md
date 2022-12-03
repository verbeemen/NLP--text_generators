# NLP - N-Gram Text Generator

The N-Gram Text Generator is a simple Python script that generates text (words) based on n-grams.  
Basically, it takes **n** characters and chooses the next character according to a probability distribution.  
(See: Markov Chain)

# How to use?

- Load the library
- Initialize the model
  - Parameters: 
    - Define the n-gram size, The amount of characters to use for predicting the next character
    - Define a path to the corpus file
  - During initialisation, the model loads its corpus and starts calculating the probability distributions for each n-gram pair

```python

    # Import the necessary libraries
    from n_gram_text_generator.generator import NGramTextGenerator

    # Initialize the generator
    generator = NGramTextGenerator(n=4, path_corpus="./data/names.csv", seed=42)
```

## Results:
 - Generating text
   - Once the model is initialised, we can easily generate text by calling the generate_text method.  
     If the n-gram parameter == 3, the model starts with 3 _start tokens_ (e.g. "###") and then generates the next character based on the probability distribution of "###". For example, the probability that the next token is 'a' is 0.25, the probability that the next token is 'b' is 0.33, c is 0.03, d is 0.05, ... z is 0.0001.   
     Then, when the model has chosen the next character (e.g. 'a'), it starts generating the next character based on the probability distribution of '##a'. This continues until the model finds its stopping criterion.   

```python
    # Show some results
    print("Results:")
    for i in range(25):
        print(f'- {generator.generate_text()}')
```
``` md
Results:
- bdelaysha
- yardelie
- aava
- carmi
- amazia
- japhea
- nashelsi
- analia
- bryer
- amuneeb
- khaleigh
- prayah
- naveah
- thoreana
- kendrick
- hazelynnlee
- eida
- rolli
- kari
- conagher
- ahlayah
- sadiq
- iris
- ...
```

Extra: It is also possible to steer the model a bit by giving it some starting characters. 

```python
    # Show some results
    print("Results:")
    for i in range(25):
        print(generator.generate_text(start="ma"))
```
```md
Results:
- maryfrankie
- maicob
- mahika
- mameda
- martrell
- marland
- maximora
- masie
- maher
- marquavius
- markiel
- manhard
- makenzie
- maxden
- maleah
- martel
- macallyssia
- makye
- mala
- maximiliannah
- ...
```

# Evaluation
How can we evaluate the model?  
A way to evaluate the model is to observe (minimize) the loss.  
And a loss can be the sum (or average) of the negative log probabilities of the generated text, a.k.a. negative log likelihood.
  
Intuitively, if the model generates a next character with a high probability, E.g. 0.8,  
then the negative log probability of that character is -log(0.8) = 0.22.
otherwise, if the model generates a next character with a very low probability, E.g. 0.01,
then the negative log probability of that character is -log(0.01) = 4.6.  
(The Negative log likelihood is always a number between 0 and infinity)