# NLP - CBOW Word Generator

TODO
(See: [https://www.youtube.com/watch?v=TCH_1BHY58I&t=3036s](https://www.youtube.com/watch?v=TCH_1BHY58I&t=3036s) 
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))

# How to use?

- TODO

```python

    # Import the necessary libraries
    from nlp_text_generators.n_gram_word_generator.text_generator import NGramWordGenerator

    # Initialize the generator
    generator = NGramWordGenerator(n=4, path_corpus="./data/names.csv", seed=42)
```

## Results:
 - Generating text
   - Once the model is initialised, we can easily generate text by calling the generate_text method.  
     TODO

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
(The Negative log likelihood is always a number between 0 and infinity.)