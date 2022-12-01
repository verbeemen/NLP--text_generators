# NLP - N-Gram Text Generator

The N-Gram Text Generator is a simple Python script that generates text (words) based on n-grams.  
Basically, it takes **n** characters and chooses the next character (semi-randomly) according to a probability distribution which was calculated from a given corpus.

# How to use?:

- Load the model
- Initialize the model
  - Parameters: 
    - Define the n-gram size, The amount of characters to use for predicting the next character
    - Define a path to a corpus file
  - During the initialization, the model will load its corpus and calculate the probability distribution of the next character given a n-gram

```python

    # Import the necessary libraries
    from n_gram_text_generator.generator import NGramTextGenerator

    # Initialize the generator
    generator = NGramTextGenerator(n=4, path_corpus="./data/names.csv", seed=42)
```

# Show Results:
 - Generate Text
   - When the model is initialized, we can easily generate text by calling the generate_text method.  
     If the n-gram parameter == 3, the model will start with 3 _start tokens_ (e.g. "###") and then it will generate the next character based on the probability distribution of '###'. E.g. the probability of the next character being 'a' is 0.25, the probability of the next character being 'b' is 0.33, c is 0.03, d is 0.05, ... z is 0.0001.   
     Next, when the model has chosen the next character (e.g. 'a'), it starts to generate the next character based on the probability distribution of '##a'. This continues until the model finds it stopping criteria.   

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

It is also possible to steer the model a little bit by providing some starting characters. 

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